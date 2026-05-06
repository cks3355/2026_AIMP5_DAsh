# 임베딩 base url 확인 필요!

# Streamlit 및 기본 라이브러리
import streamlit as st
import os

# LangChain 관련 라이브러리
# from langchain.storage import LocalFileStore
# from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
# from langchain_teddynote.prompts import load_prompt
# from langchain_teddynote import logging


# 환경 설정
from dotenv import load_dotenv

# API KEY를 환경변수로 관리하기 위한 설정 파일
load_dotenv(override=True)

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# logging.langsmith("LangChain-Tutorial")

# Streamlit 앱 제목 설정
st.title("📄 PDF 기반 QA 시스템")

# Streamlit 세션 상태 초기화 (앱 재실행 시에도 대화 기록 유지)
if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 리스트 초기화
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # RAG 체인 초기화
    st.session_state["chain"] = None

if "embeddings_initialized" not in st.session_state:
    # 임베딩 초기화 여부 추적
    st.session_state["embeddings_initialized"] = False

if "loaded_pdf_files" not in st.session_state:
    # 로드된 PDF 파일 목록
    st.session_state["loaded_pdf_files"] = []

# 사이드바 UI 구성
with st.sidebar:
    # 로드된 PDF 파일 정보 표시
    if st.session_state["loaded_pdf_files"]:
        st.info(
            f"📁 로드된 PDF 파일 ({len(st.session_state['loaded_pdf_files'])}개):\n\n"
            + "\n".join([f"• {file}" for file in st.session_state["loaded_pdf_files"]])
        )

    # PDF 재로드 버튼
    reload_btn = st.button("🔄 PDF 재로드")

    # 대화 기록 초기화 버튼
    clear_btn = st.button("🗑️ 대화 초기화")

    # 답변 길이 조절 슬라이더
    response_length = st.slider(
        "📏 답변 길이 설정",
        min_value=1,
        max_value=5,
        value=3,
        help="1: 간단 (1-2문장), 2: 짧음 (1문단), 3: 보통 (2-3문단), 4: 자세함 (4-5문단), 5: 매우 자세함 (5문단 이상)",
    )

    # 검색할 문서 개수 조절 슬라이더
    search_k = st.slider(
        "🔍 검색 문서 개수 설정",
        min_value=4,
        max_value=10,
        value=6,
        help="질문과 관련된 문서 청크를 몇 개까지 검색할지 설정합니다. 많을수록 더 많은 정보를 참고하지만 처리 시간이 길어집니다.",
    )

    # Chunk 크기 조절 슬라이더
    chunk_size = st.slider(
        "📐 Chunk 크기 설정",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="텍스트를 분할할 때 각 청크의 최대 문자 수를 설정합니다. 크기가 클수록 더 많은 문맥을 포함하지만 검색 정확도가 낮아질 수 있습니다.",
    )

    # Chunk overlap 조절 슬라이더
    chunk_overlap = st.slider(
        "🔗 Chunk Overlap 설정",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="청크 간 겹치는 문자 수를 설정합니다. 겹침이 있으면 문맥 연결성이 향상됩니다.",
    )


# 이전 대화 기록을 화면에 출력하는 함수
def print_messages():
    """저장된 대화 기록을 순서대로 화면에 표시"""
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 세션 상태에 추가하는 함수
def add_message(role, message):
    """새로운 대화 메시지를 세션 상태에 저장"""
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def format_docs(docs):
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><metadata><page>{doc.metadata['page']+1}</page><source>{doc.metadata['source']}</source></metadata></document>"
            for i, doc in enumerate(docs)
        ]
    )


# data/ 폴더의 모든 PDF 파일을 벡터 임베딩으로 변환하는 함수
def embed_pdfs_from_data_folder(chunk_size=1000, chunk_overlap=50, search_k=6):
    """data/ 폴더의 모든 PDF를 로드하여 벡터 데이터베이스 생성"""
    # 단계 1: data/ 폴더의 모든 PDF 파일 찾기
    data_folder = "./data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        st.warning("⚠️ data/ 폴더가 비어있습니다. PDF 파일을 data/ 폴더에 추가해주세요.")
        return None

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]

    if not pdf_files:
        st.warning(
            "⚠️ data/ 폴더에 PDF 파일이 없습니다. PDF 파일을 data/ 폴더에 추가해주세요."
        )
        return None

    # 단계 2: 모든 PDF 파일을 로드
    all_docs = []
    for pdf_file in pdf_files:
        file_path = os.path.join(data_folder, pdf_file)
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    # 단계 3: 문서 분할 (긴 문서를 작은 청크로 나누어 검색 성능 향상)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 각 청크의 최대 문자 수
        chunk_overlap=chunk_overlap,  # 청크 간 겹치는 문자 수 (문맥 연결성 유지)
    )
    split_documents = text_splitter.split_documents(all_docs)

    # 단계 4: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )

    # 단계 5: FAISS 벡터 데이터베이스 생성 (빠른 유사도 검색을 위함)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 6: 검색기(Retriever) 생성 (질문과 관련된 문서 청크를 찾는 역할)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": search_k}  # 설정된 개수의 관련 문서 반환
    )
    return retriever, pdf_files


# RAG 체인을 생성하는 함수 (검색-생성 파이프라인)
def create_chain(retriever, model_name="gpt-5", response_length=3):
    """Retrieval-Augmented Generation 체인 생성"""
    # 단계 6: 프롬프트 템플릿 로드
    # prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")
    prompt = ChatPromptTemplate.from_template("""
                당신은 PDF 기반 QA 도우미입니다.

                아래 문맥만 바탕으로 질문에 답하세요.
                문맥:
                {context}

                질문:
                {question}

                답변 길이 레벨:
                {response_length}

                답변:
                """)

    # 단계 7: OpenAI 언어모델 초기화 (temperature=0으로 일관된 답변 생성)
    llm = ChatOpenAI(
        model="gpt-5",
        temperature=0,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    # 단계 8: RAG 체인 구성 (검색 → 프롬프트 → LLM → 출력 파싱)
    chain = (
        {
            "context": retriever | format_docs,  # 관련 문서 검색
            "question": RunnablePassthrough(),  # 사용자 질문 전달
            "response_length": lambda _: response_length,  # 답변 길이 설정 전달
        }
        | prompt  # 프롬프트 템플릿 적용
        | llm  # 언어모델로 답변 생성
        | StrOutputParser()  # 문자열 형태로 결과 파싱
    )
    return chain


# 앱 시작 시 data/ 폴더의 PDF 파일 자동 로드
if not st.session_state["embeddings_initialized"]:
    with st.spinner("📄 data/ 폴더의 PDF 파일들을 로드하고 있습니다..."):
        result = embed_pdfs_from_data_folder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, search_k=search_k
        )

        if result is not None:
            retriever, pdf_files = result
            # RAG 체인 생성
            chain = create_chain(retriever, response_length=response_length)
            st.session_state["chain"] = chain
            st.session_state["embeddings_initialized"] = True
            st.session_state["loaded_pdf_files"] = pdf_files
            st.success(
                f"✅ {len(pdf_files)}개의 PDF 파일이 성공적으로 로드되었습니다!\n\n"
                f"📁 로드된 파일: {', '.join(pdf_files)}"
            )
        else:
            st.session_state["embeddings_initialized"] = True

# PDF 재로드 버튼 클릭 시
if reload_btn:
    st.session_state["embeddings_initialized"] = False
    st.session_state["chain"] = None
    st.session_state["messages"] = []
    st.rerun()  # 페이지 새로고침

# 대화 초기화 버튼 클릭 시
if clear_btn:
    st.session_state["messages"] = []
    st.rerun()  # 페이지 새로고침

# 이전 대화 기록 출력
print_messages()

# 사용자 질문 입력창
user_input = st.chat_input("📝 PDF 내용에 대해 궁금한 점을 물어보세요!")

# 경고 메시지 표시를 위한 빈 공간
warning_msg = st.empty()

# 사용자 질문 처리 및 답변 생성
if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자 질문 표시
        st.chat_message("user").write(user_input)
        # RAG 체인을 통해 스트리밍 답변 생성
        response = chain.stream(user_input)
        # AI 답변을 스트리밍 방식으로 실시간 표시
        with st.chat_message("assistant"):
            ai_answer = st.write_stream(response)

        # 대화 기록을 세션에 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # PDF 파일이 없을 시 경고 메시지
        warning_msg.error(
            "⚠️ data/ 폴더에 PDF 파일을 추가한 후 앱을 다시 시작해 주세요."
        )
