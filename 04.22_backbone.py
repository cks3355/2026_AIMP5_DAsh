import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict


load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_SCRIPT = BASE_DIR / "04.21_vector_db_registration.py"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5")
DEFAULT_THREAD_ID = "streamlit-multi-agent"


class RouteDecision(BaseModel):
    agent_name: str = Field(description="The best sub agent name for the current user request.")
    reason: str = Field(description="Short routing reason in Korean.")


class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    selected_agent: str
    route_reason: str
    retrieved_context: str
    term_results: list[dict[str, Any]]


@dataclass
class AgentSpec:
    name: str
    label: str
    description: str
    handler: Callable[[GraphState], dict[str, Any]]


class StandardTermEmbeddings(OpenAIEmbeddings):
    """기존 프로젝트의 환경 변수 규칙을 그대로 따르는 임베딩 래퍼입니다."""

    def __init__(self) -> None:
        super().__init__(
            model="text-embedding-3-large",
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        )


def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=temperature,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )


def load_vector_db_module():
    spec = importlib.util.spec_from_file_location("vector_db_registration", VECTOR_DB_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("04.21_vector_db_registration.py 파일을 불러오지 못했습니다.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def get_standard_term_vectorstore():
    module = load_vector_db_module()
    documents = module.build_standard_term_documents()
    return module.register_terms_to_vector_db(documents)


def search_standard_terms(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    vectorstore = get_standard_term_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    formatted_results: list[dict[str, Any]] = []
    for document, distance in results:
        formatted_results.append(
            {
                "standard_term_name": document.metadata.get("standard_term_name", "알 수 없음"),
                "domain": document.metadata.get("domain", "미분류"),
                "description": extract_description(document),
                "distance": float(distance),
                "similarity": 1 / (1 + float(distance)),
            }
        )
    return formatted_results


def extract_description(document: Document) -> str:
    for line in document.page_content.splitlines():
        normalized = line.strip()
        if normalized.startswith("설명:"):
            return normalized.replace("설명:", "", 1).strip()
    return document.page_content.strip()


@st.cache_resource(show_spinner=False)
def get_pdf_retriever(chunk_size: int = 1200, chunk_overlap: int = 120):
    if not DATA_DIR.exists():
        return None, []

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return None, []

    all_docs: list[Document] = []
    for pdf_file in pdf_files:
        loader = PDFPlumberLoader(str(pdf_file))
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_documents = splitter.split_documents(all_docs)

    embeddings = StandardTermEmbeddings()
    vectorstore = FAISS.from_documents(split_documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever, [pdf_file.name for pdf_file in pdf_files]


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "참고 문서를 찾지 못했습니다."

    formatted: list[str] = []
    for index, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        source = Path(str(doc.metadata.get("source", "unknown"))).name
        page_label = page + 1 if isinstance(page, int) else "?"
        formatted.append(
            f"[문서 {index}] source={source}, page={page_label}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted)


def build_agent_registry(llm: ChatOpenAI) -> dict[str, AgentSpec]:
    return {
        "general_agent": AgentSpec(
            name="general_agent",
            label="General Agent",
            description="일반적인 질의응답, 아키텍처 설명, 멀티 에이전트 설계 안내를 담당합니다.",
            handler=lambda state: run_general_agent(llm, state),
        ),
        "pdf_qa_agent": AgentSpec(
            name="pdf_qa_agent",
            label="PDF QA Agent",
            description="data 폴더의 PDF 문서를 근거로 질문에 답합니다.",
            handler=lambda state: run_pdf_qa_agent(llm, state),
        ),
        "standard_term_agent": AgentSpec(
            name="standard_term_agent",
            label="Standard Term Agent",
            description="표준용어 벡터 DB를 검색해 가까운 용어와 설명을 안내합니다.",
            handler=lambda state: run_standard_term_agent(llm, state),
        ),
    }


def build_supervisor_prompt(registry: dict[str, AgentSpec]) -> str:
    agent_guide = "\n".join(
        f"- {agent.name}: {agent.description}"
        for agent in registry.values()
    )
    return f"""
당신은 Super Agent입니다.
사용자 요청을 보고 가장 적합한 Sub Agent 하나를 선택하세요.

[최근 1주일 프로젝트 맥락]
- 04.20: PDF 기반 QA 실험
- 04.21: FAISS 기반 표준용어 벡터 검색
- 04.22: Streamlit UI와 멀티 에이전트 백본 설계

[선택 가능한 Sub Agent]
{agent_guide}

[라우팅 규칙]
1. PDF, 문서 근거, 사용자 가이드, 자료 요약 요청은 pdf_qa_agent를 우선 선택합니다.
2. 표준용어, 용어 검색, 유사 용어, 도메인 용어 비교 요청은 standard_term_agent를 우선 선택합니다.
3. 아키텍처 설명, 일반 대화, Streamlit/LangGraph 구현 안내는 general_agent를 선택합니다.
4. agent_name은 반드시 위 목록 중 하나만 반환합니다.
5. reason은 1문장 한국어로 짧게 작성합니다.
""".strip()


def get_last_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def route_with_fallback(user_message: str, registry: dict[str, AgentSpec]) -> RouteDecision:
    lowered = user_message.lower()
    if any(keyword in user_message for keyword in ["PDF", "문서", "가이드", "자료", "사용자 가이드"]):
        return RouteDecision(agent_name="pdf_qa_agent", reason="문서 근거가 필요한 질문으로 판단했습니다.")
    if any(keyword in user_message for keyword in ["표준용어", "용어", "유사", "도메인"]):
        return RouteDecision(agent_name="standard_term_agent", reason="표준용어 검색이 필요한 요청으로 판단했습니다.")
    if "langgraph" in lowered or "streamlit" in lowered or "agent" in lowered:
        return RouteDecision(agent_name="general_agent", reason="아키텍처 또는 구현 설명 요청으로 판단했습니다.")
    first_agent = next(iter(registry))
    return RouteDecision(agent_name=first_agent, reason="기본 응답 에이전트로 처리합니다.")


def make_supervisor_node(llm: ChatOpenAI, registry: dict[str, AgentSpec]):
    router = llm.with_structured_output(RouteDecision)
    system_prompt = build_supervisor_prompt(registry)

    def supervisor(state: GraphState) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_message = get_last_user_message(messages)
        decision: RouteDecision

        try:
            decision = router.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
            )
            if decision.agent_name not in registry:
                decision = route_with_fallback(user_message, registry)
        except Exception:
            decision = route_with_fallback(user_message, registry)

        return {
            "selected_agent": decision.agent_name,
            "route_reason": decision.reason,
        }

    return supervisor


def run_general_agent(llm: ChatOpenAI, state: GraphState) -> dict[str, Any]:
    messages = state.get("messages", [])
    system_prompt = SystemMessage(
        content="""
당신은 Streamlit 기반 멀티 에이전트 챗봇의 General Agent입니다.
사용자 질문에 친절하고 실무적으로 답변하세요.
특히 LangGraph 구조 설명, Streamlit UI 구성, Supervisor/Sub Agent 설계 원칙을 명확하게 설명하세요.
""".strip()
    )
    response = llm.invoke([system_prompt, *messages])
    return {"messages": [AIMessage(content=response.content)]}


def run_pdf_qa_agent(llm: ChatOpenAI, state: GraphState) -> dict[str, Any]:
    retriever, pdf_files = get_pdf_retriever()
    user_message = get_last_user_message(state.get("messages", []))

    if retriever is None:
        return {
            "retrieved_context": "",
            "messages": [
                AIMessage(
                    content="`data/` 폴더에 PDF가 없어 PDF QA Agent를 실행할 수 없습니다. PDF 파일을 추가한 뒤 다시 시도해주세요."
                )
            ],
        }

    docs = retriever.invoke(user_message)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 PDF QA Agent입니다.
반드시 제공된 문맥을 우선 근거로 답변하세요.
문맥에 없는 내용은 추정하지 말고, 부족한 점을 명확하게 말하세요.
답변 말미에 참고한 PDF 파일명을 간단히 덧붙이세요.
""".strip(),
            ),
            (
                "human",
                """
[질문]
{question}

[문맥]
{context}

[PDF 목록]
{pdf_files}
""".strip(),
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {
            "question": user_message,
            "context": context,
            "pdf_files": ", ".join(pdf_files),
        }
    )
    return {
        "retrieved_context": context,
        "messages": [AIMessage(content=answer)],
    }


def run_standard_term_agent(llm: ChatOpenAI, state: GraphState) -> dict[str, Any]:
    user_message = get_last_user_message(state.get("messages", []))
    term_results = search_standard_terms(user_message)

    if not term_results:
        return {
            "term_results": [],
            "messages": [
                AIMessage(content="관련 표준용어를 찾지 못했습니다. 조금 더 구체적인 설명으로 다시 요청해주세요.")
            ],
        }

    context_lines = []
    for index, item in enumerate(term_results, start=1):
        context_lines.append(
            f"{index}. 용어명={item['standard_term_name']}, domain={item['domain']}, "
            f"similarity={item['similarity']:.4f}, 설명={item['description']}"
        )
    context = "\n".join(context_lines)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 Standard Term Agent입니다.
유사한 표준용어 후보를 요약하고, 어떤 용어가 왜 가까운지 설명하세요.
가능하면 가장 적합한 후보 1개를 먼저 제시하고 나머지 후보는 비교 형태로 정리하세요.
""".strip(),
            ),
            (
                "human",
                """
[사용자 요청]
{question}

[검색 결과]
{context}
""".strip(),
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": user_message, "context": context})
    return {
        "term_results": term_results,
        "messages": [AIMessage(content=answer)],
    }


def build_graph():
    llm = get_llm()
    registry = build_agent_registry(llm)
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("supervisor", make_supervisor_node(llm, registry))
    for agent_name, agent in registry.items():
        graph_builder.add_node(agent_name, agent.handler)

    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("selected_agent"),
        {agent_name: agent_name for agent_name in registry},
    )
    for agent_name in registry:
        graph_builder.add_edge(agent_name, END)

    graph = graph_builder.compile(checkpointer=MemorySaver())
    return graph, registry


@st.cache_resource(show_spinner=False)
def get_graph_resources():
    return build_graph()


def render_sidebar(registry: dict[str, AgentSpec], pdf_files: list[str]) -> None:
    with st.sidebar:
        st.subheader("Agent Topology")
        st.markdown("**Super Agent**")
        st.caption("사용자 요청을 분석해 가장 적합한 Sub Agent로 라우팅합니다.")

        for agent in registry.values():
            with st.container(border=True):
                st.markdown(f"**{agent.label}**")
                st.caption(agent.description)

        st.divider()
        st.subheader("Recent Context")
        st.caption("최근 1주일 코드 흐름을 반영했습니다.")
        st.write("- PDF 기반 QA")
        st.write("- 표준용어 벡터 검색")
        st.write("- Streamlit UI 실험")

        st.divider()
        st.subheader("Data")
        if pdf_files:
            st.success(f"PDF {len(pdf_files)}개 연결됨")
            for pdf_file in pdf_files:
                st.write(f"- {pdf_file}")
        else:
            st.warning("연결된 PDF가 없습니다.")

        if st.button("대화 초기화", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.session_state["route_history"] = []
            st.rerun()


def ensure_session_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "route_history" not in st.session_state:
        st.session_state["route_history"] = []
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = f"{DEFAULT_THREAD_ID}-{uuid4()}"


def render_chat_history() -> None:
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def append_chat_message(role: str, content: str) -> None:
    st.session_state["chat_messages"].append({"role": role, "content": content})


def to_langchain_messages() -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for message in st.session_state["chat_messages"]:
        if message["role"] == "user":
            converted.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            converted.append(AIMessage(content=message["content"]))
    return converted


def main() -> None:
    st.set_page_config(
        page_title="04.22 Multi Agent Backbone",
        page_icon="🤖",
        layout="wide",
    )
    ensure_session_state()

    graph, registry = get_graph_resources()
    _, pdf_files = get_pdf_retriever()

    st.title("04.22 Multi Agent Backbone")
    st.caption("Streamlit UI + LangGraph Supervisor/Sub Agent 구조로 확장 가능한 AI 챗봇 백본입니다.")

    render_sidebar(registry, pdf_files)

    top_col, info_col = st.columns([2, 1], gap="large")
    with top_col:
        st.markdown(
            """
이 화면은 `Super Agent 1개 + Sub Agent 여러 개` 구조를 기준으로 동작합니다.
새로운 Sub Agent는 레지스트리에 추가하면 그래프 노드와 라우팅 후보에 자동 포함되도록 설계했습니다.
"""
        )
    with info_col:
        last_route = st.session_state["route_history"][-1] if st.session_state["route_history"] else None
        with st.container(border=True):
            st.markdown("**Last Routing**")
            if last_route:
                st.write(f"Agent: `{last_route['agent']}`")
                st.write(last_route["reason"])
            else:
                st.caption("아직 라우팅 기록이 없습니다.")

    render_chat_history()

    user_input = st.chat_input("질문을 입력하세요. 예: 표준용어 추천, PDF 요약, 멀티 에이전트 구조 설명")
    if not user_input:
        return

    append_chat_message("user", user_input)
    with st.chat_message("user"):
        st.write(user_input)

    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    graph_input: GraphState = {"messages": to_langchain_messages()}

    with st.chat_message("assistant"):
        with st.spinner("Super Agent가 적절한 Sub Agent를 선택하고 있습니다..."):
            result = graph.invoke(graph_input, config=config)

        assistant_messages = result.get("messages", [])
        assistant_reply = ""
        if assistant_messages:
            last_message = assistant_messages[-1]
            assistant_reply = last_message.content if isinstance(last_message.content, str) else str(last_message.content)
        else:
            assistant_reply = "응답을 생성하지 못했습니다."

        selected_agent = result.get("selected_agent", "unknown")
        route_reason = result.get("route_reason", "라우팅 사유를 확인하지 못했습니다.")

        st.caption(f"Routed to: {selected_agent}")
        st.write(assistant_reply)

    append_chat_message("assistant", assistant_reply)
    st.session_state["route_history"].append({"agent": selected_agent, "reason": route_reason})


if __name__ == "__main__":
    main()
