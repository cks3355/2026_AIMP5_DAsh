import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from openai import AzureOpenAI


load_dotenv(override=True)


class AzureOpenAIEmbeddings(Embeddings):
    """Azure OpenAI embeddings API를 LangChain FAISS에 연결하기 위한 래퍼입니다."""

    def __init__(self) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("EMBEDDING_BASE_URL", "https://aitalentlab.skax.co.kr"),
            api_key=os.getenv("LLM_API_KEY"),
            api_version="2024-12-01-preview",
        )
        self.model = "text-embedding-3-large"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding


def build_standard_term_documents() -> list[Document]:
    """표준용어명과 설명을 벡터 DB에 넣기 좋은 Document 형태로 변환합니다."""
    standard_terms = [
        {
            "standard_term_name": "고객ID",
            "description": "고객을 고유하게 식별하는 내부 관리용 식별자입니다.",
            "domain": "고객",
        },
        {
            "standard_term_name": "계좌번호",
            "description": "금융 거래 계좌를 구분하기 위해 부여된 번호입니다.",
            "domain": "계좌",
        },
        {
            "standard_term_name": "거래일자",
            "description": "금융 거래가 실제로 발생한 기준 일자입니다.",
            "domain": "거래",
        },
        {
            "standard_term_name": "상품코드",
            "description": "상품이나 서비스를 식별하기 위해 정의한 표준 코드입니다.",
            "domain": "상품",
        },
    ]

    documents = []
    for term in standard_terms:
        page_content = (
            f"표준용어명: {term['standard_term_name']}\n"
            f"설명: {term['description']}"
        )
        metadata = {
            "standard_term_name": term["standard_term_name"],
            "domain": term["domain"],
            "source": "standard_terms_example",
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def create_embeddings() -> AzureOpenAIEmbeddings:
    """AzureOpenAI 임베딩 클라이언트를 생성합니다."""
    return AzureOpenAIEmbeddings()


def register_terms_to_vector_db(documents: list[Document]) -> FAISS:
    """문서를 임베딩하여 FAISS 벡터 DB에 등록합니다."""
    embeddings = create_embeddings()
    vectorstore = FAISS.from_documents(
        documents=[documents[0]],
        embedding=embeddings,
    )

    remaining_documents = documents[1:]
    if remaining_documents:
        ids = [str(uuid4()) for _ in remaining_documents]
        vectorstore.add_documents(documents=remaining_documents, ids=ids)

    return vectorstore


def search_similar_term(vectorstore: FAISS, query: str) -> None:
    """등록된 용어 중 질의와 유사한 항목을 조회합니다."""
    results = vectorstore.similarity_search_with_score(query, k=3)

    print(f"\n[유사도 검색 질의] {query}")
    for index, (doc, score) in enumerate(results, start=1):
        print(f"\n[{index}] score={score:.4f}")
        print(doc.page_content)
        print(f"metadata={doc.metadata}")


def main() -> None:
    documents = build_standard_term_documents()
    vectorstore = register_terms_to_vector_db(documents)

    print(f"총 {len(documents)}건의 표준용어를 벡터 DB에 등록했습니다.")
    search_similar_term(vectorstore, "고객을 식별하는 번호")


if __name__ == "__main__":
    main()
