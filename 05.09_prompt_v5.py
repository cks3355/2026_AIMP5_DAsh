from __future__ import annotations

# DAsh: Pi-mono Super Agent routing + LangGraph Sub Agent prototype.
#
# 사용자가 전달한 Pi-mono 예시의 핵심 원칙을 Python으로 반영했습니다.
# - 내부 대화는 AgentMessage로 유지하고 LLM 호출 경계에서만 provider message로 변환합니다.
# - Super Agent는 Pi-mono 스타일 event stream(SSE 로그), turn/memory, routing을 관리합니다.
# - Sub Agent는 LangGraph node/edge(select/recommend/validate/change)로 구성합니다.
# - Mock RDB/Vector DB로 바로 실행되며, .env 설정으로 Oracle/FAISS/OpenAI 교체 지점을 둡니다.
#
# 실행 방법:
#     streamlit run 05.09_prompt_v5.py
#     python 05.09_prompt_v5.py "작업이 시작된 일시"

import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel, Field

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    MemorySaver = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except Exception:  # pragma: no cover
    TavilySearchResults = None  # type: ignore[assignment]

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[misc,assignment]


BASE_DIR = Path(__file__).resolve().parent
SIMILARITY_TOP_K = 3
SIMILARITY_THRESHOLD = 0.6


class Intent(StrEnum):
    """명세에서 요구한 사용자 intent입니다."""

    MEANING_SELECT = "meaning_select"
    TERM_RECOMMEND = "term_recommend"
    TERM_INSERT = "term_insert"
    TERM_UPDATE = "term_update"
    TERM_DELETE = "term_delete"
    DA_REQUEST = "da_request"


ChangeAction = Literal["insert", "update", "delete"]


class AgentMessage(BaseModel):
    """Pi-mono 내부에서 유지하는 메시지입니다."""

    role: Literal["system", "user", "assistant", "toolResult"]
    content: str
    name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class AgentEvent(BaseModel):
    """Pi-mono SSE 스타일 이벤트입니다."""

    type: str
    agent: str
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class RouterDecision(BaseModel):
    """Super Agent routing 단계가 JSON으로 유지하는 분류 결과입니다."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    entities: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: str | None = None


class StandardTerm(BaseModel):
    """RDB와 Vector DB에서 공유하는 표준용어입니다."""

    logical_name: str
    description: str
    physical_name: str | None = None
    domain: str = "common"
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class StandardWord(BaseModel):
    """RDB에서 조회되는 표준단어입니다."""

    logical_name: str
    physical_name: str
    meaning_hint: str


class SimilarTerm(BaseModel):
    """Vector DB 의미 검색 결과입니다."""

    term: StandardTerm
    similarity: float


class Recommendation(BaseModel):
    """추천 표준용어 후보입니다."""

    logical_name: str
    description: str
    reason: str
    score: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """validate_agent의 검증 결과입니다."""

    ok: bool
    reason: str
    normalized_term: StandardTerm | None = None


class ChangeRequest(BaseModel):
    """change_agent에 전달되는 변경 요청입니다."""

    action: ChangeAction
    logical_name: str
    description: str | None = None
    new_logical_name: str | None = None


class AgentTurnState(BaseModel):
    """대화 턴 관리를 위한 LangGraph 상태입니다."""

    session_id: str
    user_input: str
    messages: list[AgentMessage] = Field(default_factory=list)
    memory: list[AgentMessage] = Field(default_factory=list)
    lookup_count: int = 0
    router_decision: RouterDecision | None = None
    similar_terms: list[SimilarTerm] = Field(default_factory=list)
    standard_words: list[StandardWord] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    validation: ValidationResult | None = None
    change_request: ChangeRequest | None = None
    final_answer: str = ""
    logs: list[AgentEvent] = Field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class AppConfig:
    """환경 변수 기반 실행 설정입니다."""

    mock_mode: bool
    openai_api_key: str | None
    tavily_api_key: str | None
    oracle_url: str | None
    langsmith_project: str | None
    model_name: str = "gpt-5.4"
    embedding_model_name: str = "text-embedding-3-large"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """DB/API 연결 정보를 .env에서 읽습니다."""

        load_dotenv(BASE_DIR / ".env")
        return cls(
            mock_mode=os.getenv("MOCK_MODE", "true").lower() != "false",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            oracle_url=os.getenv("ORACLE_URL") or os.getenv("DATABASE_URL"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT"),
            model_name=os.getenv("OPENAI_MODEL", "gpt-5.4"),
            embedding_model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        )


def event(event_type: str, agent: str, message: str = "", **payload: Any) -> AgentEvent:
    """AgentEvent를 간결하게 생성합니다."""

    return AgentEvent(type=event_type, agent=agent, message=message, payload=payload)


class EventStream:
    """Pi-mono EventStream의 Python 축약 구현입니다."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.result: AgentTurnState | None = None

    def push(self, item: AgentEvent) -> None:
        """이벤트를 스트림에 추가합니다."""

        self.events.append(item)

    def end(self, result: AgentTurnState) -> None:
        """최종 상태를 저장합니다."""

        self.result = result

    def sse(self) -> list[str]:
        """UI 하단에 표시할 SSE 라인을 반환합니다."""

        return [f"event: {item.type}\ndata: {item.model_dump_json()}" for item in self.events]


class EmbeddingService(Protocol):
    """Embedding 교체를 위한 인터페이스입니다."""

    def embed(self, value: str) -> list[float]:
        """텍스트를 벡터로 변환합니다."""


class HashEmbeddingService:
    """Mock 모드용 deterministic embedding입니다."""

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, value: str) -> list[float]:
        """문자 n-gram hash를 사용해 cosine 검색용 벡터를 생성합니다."""

        vector = [0.0] * self.dimensions
        compact = re.sub(r"\s+", "", value.lower())
        tokens = [compact[i : i + 2] for i in range(max(1, len(compact) - 1))]
        tokens += re.findall(r"[가-힣A-Za-z0-9]+", value.lower())
        for token in tokens:
            slot = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:2], "big") % self.dimensions
            vector[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbeddingService:
    """운영 모드에서 text-embedding-3-large를 호출합니다."""

    def __init__(self, config: AppConfig) -> None:
        if OpenAIEmbeddings is None:
            raise RuntimeError("langchain-openai가 설치되어 있지 않습니다.")
        self.client = OpenAIEmbeddings(model=config.embedding_model_name, api_key=config.openai_api_key)

    def embed(self, value: str) -> list[float]:
        """OpenAI embedding API를 호출합니다."""

        return cast(list[float], self.client.embed_query(value))


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity를 계산합니다."""

    size = min(len(left), len(right))
    if size == 0:
        return 0.0
    dot = sum(left[i] * right[i] for i in range(size))
    ln = math.sqrt(sum(v * v for v in left[:size])) or 1.0
    rn = math.sqrt(sum(v * v for v in right[:size])) or 1.0
    return dot / (ln * rn)


class TermRepository(Protocol):
    """RDB 접근 인터페이스입니다."""

    def list_terms(self) -> list[StandardTerm]: ...
    def list_words(self) -> list[StandardWord]: ...
    def get_term(self, logical_name: str) -> StandardTerm | None: ...
    def upsert_term(self, term: StandardTerm) -> None: ...
    def delete_term(self, logical_name: str) -> None: ...


class MockTermRepository:
    """Oracle 없이 실행 가능한 Mock RDB입니다."""

    def __init__(self) -> None:
        self.terms = {
            "작업시작일시": StandardTerm(logical_name="작업시작일시", physical_name="JOB_STRT_DTTM", description="작업이 start된 시점", domain="manufacturing"),
            "작업완료일시": StandardTerm(logical_name="작업완료일시", physical_name="JOB_CMPL_DTTM", description="작업 처리가 완료된 일시", domain="manufacturing"),
            "재공수량": StandardTerm(logical_name="재공수량", physical_name="WIP_QTY", description="제조 공정에서 생산 중이거나 가공 중인 미완성 제품의 수량", domain="manufacturing"),
            "표준체크리스트": StandardTerm(logical_name="표준체크리스트", physical_name="STD_CHK_LIST", description="데이터 표준 준수 여부를 점검하기 위한 항목 목록", domain="governance"),
        }
        self.words = [
            StandardWord(logical_name="작업", physical_name="JOB", meaning_hint="업무나 배치 처리 단위"),
            StandardWord(logical_name="시작", physical_name="STRT", meaning_hint="처리가 처음 개시됨"),
            StandardWord(logical_name="종료", physical_name="END", meaning_hint="처리가 끝남"),
            StandardWord(logical_name="완료", physical_name="CMPL", meaning_hint="정상적으로 마무리됨"),
            StandardWord(logical_name="일시", physical_name="DTTM", meaning_hint="날짜와 시각"),
            StandardWord(logical_name="수량", physical_name="QTY", meaning_hint="대상의 개수"),
            StandardWord(logical_name="재공", physical_name="WIP", meaning_hint="가공 중인 미완성 제품"),
            StandardWord(logical_name="표준", physical_name="STD", meaning_hint="조직에서 합의한 기준"),
            StandardWord(logical_name="체크리스트", physical_name="CHK_LIST", meaning_hint="점검 항목 목록"),
        ]

    def list_terms(self) -> list[StandardTerm]:
        """전체 표준용어를 반환합니다."""

        return list(self.terms.values())

    def list_words(self) -> list[StandardWord]:
        """전체 표준단어를 반환합니다."""

        return list(self.words)

    def get_term(self, logical_name: str) -> StandardTerm | None:
        """논리명으로 표준용어를 조회합니다."""

        return self.terms.get(logical_name)

    def upsert_term(self, term: StandardTerm) -> None:
        """표준용어를 등록 또는 수정합니다."""

        self.terms[term.logical_name] = term

    def delete_term(self, logical_name: str) -> None:
        """표준용어를 삭제합니다."""

        if logical_name not in self.terms:
            raise KeyError(f"{logical_name} 표준용어가 없습니다.")
        del self.terms[logical_name]


class OracleTermRepository:
    """SQLAlchemy Oracle 교체 구현입니다."""

    def __init__(self, oracle_url: str) -> None:
        if create_engine is None or text is None:
            raise RuntimeError("SQLAlchemy가 설치되어 있지 않습니다.")
        self.engine = create_engine(oracle_url)

    def list_terms(self) -> list[StandardTerm]:
        """Oracle에서 표준용어를 조회합니다."""

        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT logical_name, physical_name, description, domain FROM standard_terms WHERE deleted_yn='N'")).mappings()
            return [StandardTerm(**dict(row)) for row in rows]

    def list_words(self) -> list[StandardWord]:
        """Oracle에서 표준단어를 조회합니다."""

        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT logical_name, physical_name, meaning_hint FROM standard_words WHERE use_yn='Y'")).mappings()
            return [StandardWord(**dict(row)) for row in rows]

    def get_term(self, logical_name: str) -> StandardTerm | None:
        """Oracle에서 단건 표준용어를 조회합니다."""

        with self.engine.begin() as conn:
            row = conn.execute(text("SELECT logical_name, physical_name, description, domain FROM standard_terms WHERE logical_name=:n AND deleted_yn='N'"), {"n": logical_name}).mappings().first()
            return StandardTerm(**dict(row)) if row else None

    def upsert_term(self, term: StandardTerm) -> None:
        """Oracle에 표준용어를 MERGE합니다."""

        sql = """
        MERGE INTO standard_terms t
        USING (SELECT :logical_name logical_name, :physical_name physical_name, :description description, :domain domain FROM dual) s
        ON (t.logical_name=s.logical_name)
        WHEN MATCHED THEN UPDATE SET t.physical_name=s.physical_name, t.description=s.description, t.domain=s.domain, t.deleted_yn='N', t.updated_at=SYSTIMESTAMP
        WHEN NOT MATCHED THEN INSERT (logical_name, physical_name, description, domain, deleted_yn, updated_at)
        VALUES (s.logical_name, s.physical_name, s.description, s.domain, 'N', SYSTIMESTAMP)
        """
        with self.engine.begin() as conn:
            conn.execute(text(sql), term.model_dump())

    def delete_term(self, logical_name: str) -> None:
        """Oracle 표준용어를 논리 삭제합니다."""

        with self.engine.begin() as conn:
            conn.execute(text("UPDATE standard_terms SET deleted_yn='Y', updated_at=SYSTIMESTAMP WHERE logical_name=:n"), {"n": logical_name})


class VectorRepository(Protocol):
    """Vector DB 접근 인터페이스입니다."""

    def search(self, description: str, top_k: int = SIMILARITY_TOP_K) -> list[SimilarTerm]: ...
    def upsert(self, term: StandardTerm) -> None: ...
    def delete(self, logical_name: str) -> None: ...


class MockVectorRepository:
    """FAISS 없이 동작하는 메모리 Vector DB입니다."""

    def __init__(self, terms: TermRepository, embedder: EmbeddingService) -> None:
        self.embedder = embedder
        self.rows: dict[str, tuple[StandardTerm, list[float]]] = {}
        for term in terms.list_terms():
            self.upsert(term)

    def search(self, description: str, top_k: int = SIMILARITY_TOP_K) -> list[SimilarTerm]:
        """similarity 0.6 이상 상위 3개 표준용어를 반환합니다."""

        query = self.embedder.embed(description)
        ranked = [SimilarTerm(term=term, similarity=cosine(query, vec)) for term, vec in self.rows.values()]
        ranked.sort(key=lambda x: x.similarity, reverse=True)
        return [item for item in ranked[:top_k] if item.similarity >= SIMILARITY_THRESHOLD]

    def upsert(self, term: StandardTerm) -> None:
        """용어 설명 embedding을 저장합니다."""

        self.rows[term.logical_name] = (term, self.embedder.embed(f"{term.logical_name} {term.description}"))

    def delete(self, logical_name: str) -> None:
        """Vector DB에서 용어를 제거합니다."""

        self.rows.pop(logical_name, None)


class FaissVectorRepository(MockVectorRepository):
    """FAISS 연결 교체 지점입니다. 현재는 실행성을 위해 Mock 구현을 상속합니다."""


@dataclass(frozen=True)
class AgentTool:
    """Pi-mono AgentTool과 유사한 tool 정의입니다."""

    name: str
    description: str
    execute: Callable[..., Any]
    execution_mode: Literal["parallel", "sequential"] = "parallel"


@dataclass
class ToolBox:
    """Sub Agent가 호출하는 tool 묶음입니다."""

    term_repository: TermRepository
    vector_repository: VectorRepository
    config: AppConfig
    tools: dict[str, AgentTool] = field(init=False)

    def __post_init__(self) -> None:
        self.tools = {
            "search_similar_terms": AgentTool("search_similar_terms", "Vector DB 유사 용어 조회", self.search_similar_terms),
            "list_standard_words": AgentTool("list_standard_words", "RDB 표준단어 조회", self.list_standard_words),
            "change_standard_term": AgentTool("change_standard_term", "RDB/Vector DB 변경", self.change_standard_term, "sequential"),
            "search_reference": AgentTool("search_reference", "RDB/Vector/웹 문의 검색", self.search_reference),
        }

    def search_similar_terms(self, description: str) -> list[SimilarTerm]:
        """Vector DB에서 유사 표준용어를 조회합니다."""

        return self.vector_repository.search(description)

    def list_standard_words(self) -> list[StandardWord]:
        """RDB에서 표준단어를 조회합니다."""

        return self.term_repository.list_words()

    def change_standard_term(self, request: ChangeRequest, term: StandardTerm | None) -> str:
        """검증된 변경을 RDB와 Vector DB에 반영합니다."""

        if request.action == "delete":
            self.term_repository.delete_term(request.logical_name)
            self.vector_repository.delete(request.logical_name)
            return "변경이 완료되었습니다."
        if term is None:
            raise ValueError("검증된 표준용어가 없습니다.")
        if request.action == "update" and request.new_logical_name and request.new_logical_name != request.logical_name:
            self.term_repository.delete_term(request.logical_name)
            self.vector_repository.delete(request.logical_name)
        self.term_repository.upsert_term(term)
        self.vector_repository.upsert(term)
        return "등록이 완료되었습니다." if request.action == "insert" else "변경이 완료되었습니다."

    def search_reference(self, question: str) -> str:
        """문의 사항에 대해 RDB, Vector DB, 선택적 Tavily 결과를 요약합니다."""

        words = [f"{w.logical_name}: {w.meaning_hint}" for w in self.term_repository.list_words() if w.logical_name in question]
        terms = [f"{r.term.logical_name}: {r.term.description}" for r in self.vector_repository.search(question)]
        web: list[str] = []
        if not self.config.mock_mode and self.config.tavily_api_key and TavilySearchResults:
            try:
                web = [str(x.get("content", ""))[:160] for x in TavilySearchResults(max_results=2, api_key=self.config.tavily_api_key).invoke(question)]
            except Exception as exc:
                web = [f"웹 검색 실패: {exc}"]
        evidence = words + terms + web
        return "\n".join(evidence[:5]) if evidence else "관련 표준 정보를 찾지 못했습니다. 업무 맥락을 더 구체적으로 입력해 주세요."


class LlmBoundary:
    """AgentMessage[]를 LLM provider message로 변환하는 Pi-mono 경계입니다."""

    def __init__(self, config: AppConfig) -> None:
        self.client = None
        if not config.mock_mode and ChatOpenAI is not None:
            self.client = ChatOpenAI(model=config.model_name, api_key=config.openai_api_key, temperature=0.1)

    def convert_to_llm(self, messages: list[AgentMessage]) -> list[dict[str, str]]:
        """AgentMessage를 provider role/content로 변환합니다."""

        return [{"role": "tool" if m.role == "toolResult" else m.role, "content": m.content} for m in messages]


def ambiguous(value: str) -> bool:
    """설명이 너무 짧거나 모호한지 판단합니다."""

    return len(value.strip()) < 6 or value.strip() in {"재공", "일시", "수량", "상태", "코드"}


def extract_entities(value: str) -> dict[str, Any]:
    """한국어 변경 문장에서 논리명과 설명을 추출합니다."""

    entities: dict[str, Any] = {}
    quoted = re.findall(r"['\"`]([^'\"`]+)['\"`]", value)
    if quoted:
        entities["logical_name"] = quoted[0]
    if len(quoted) >= 2:
        entities["description"] = quoted[-1]
    elif "신청:" in value:
        entities["description"] = value.split("신청:", 1)[1].strip()
    return entities


def physical_name(logical_name: str, words: list[StandardWord]) -> str:
    """표준단어 물리명을 조합해 물리명 후보를 생성합니다."""

    parts = [w.physical_name for w in words if w.logical_name in logical_name]
    return "_".join(parts) if parts else "TERM_" + hashlib.sha1(logical_name.encode("utf-8")).hexdigest()[:8].upper()


class RouterSkill:
    """사용자 intent를 JSON으로 분류하는 skill입니다."""

    def classify(self, value: str, lookup_count: int) -> RouterDecision:
        """명세의 6개 intent 중 하나로 분류합니다."""

        entities = extract_entities(value)
        if any(k in value for k in ["삭제", "제거"]):
            return RouterDecision(intent=Intent.TERM_DELETE, confidence=0.9, rationale="삭제 지시어", entities=entities)
        if any(k in value for k in ["변경", "수정", "바꿔"]):
            return RouterDecision(intent=Intent.TERM_UPDATE, confidence=0.9, rationale="수정 지시어", entities=entities)
        if any(k in value for k in ["신청", "등록", "추가"]) and entities.get("logical_name"):
            return RouterDecision(intent=Intent.TERM_INSERT, confidence=0.86, rationale="등록 지시어", entities=entities)
        if "추천" in value or (lookup_count > 0 and value.strip() == "추천"):
            return RouterDecision(intent=Intent.TERM_RECOMMEND, confidence=0.84, rationale="추천 요청", entities=entities)
        if value.endswith("?") or any(k in value for k in ["이란", "뭔가", "체크리스트", "가이드"]):
            return RouterDecision(intent=Intent.DA_REQUEST, confidence=0.78, rationale="문의형 입력", entities=entities)
        if ambiguous(value):
            return RouterDecision(intent=Intent.MEANING_SELECT, confidence=0.62, rationale="설명 조회이나 모호함", entities=entities, needs_clarification=True, clarification_question="대상, 상태, 속성을 포함해 한 문장으로 더 설명해 주세요.")
        return RouterDecision(intent=Intent.MEANING_SELECT, confidence=0.82, rationale="용어 설명 입력", entities=entities)


class RecommendSkill:
    """Tree of Thoughts와 Self-Correction으로 추천 후보를 생성하는 skill입니다."""

    def recommend(self, description: str, words: list[StandardWord], existing: list[StandardTerm]) -> list[Recommendation]:
        """여러 조합 경로를 만들고 자체 보정 후 상위 후보를 반환합니다."""

        matched = [w for w in words if w.logical_name in description or any(t in description for t in w.meaning_hint.split())]
        for must in ["작업", "종료", "일시", "재공", "수량", "표준", "체크리스트"]:
            if must in description and all(w.logical_name != must for w in matched):
                matched += [w for w in words if w.logical_name == must]
        matched = matched or words[:3]
        paths = [matched[:4], sorted(matched[:4], key=lambda w: 0 if w.logical_name in {"작업", "재공", "표준"} else 1), [w for w in matched[:4] if w.logical_name != "완료"] or matched[:4]]
        existing_names = {t.logical_name for t in existing}
        results: dict[str, Recommendation] = {}
        for index, path in enumerate(paths):
            name = "".join(w.logical_name for w in path)
            reason = f"ToT 경로 {index + 1}: " + ", ".join(w.logical_name for w in path)
            score = 0.9 - index * 0.08
            if "일시" in description and "일시" not in name:
                name += "일시"
                reason += "; Self-Correction: 시간 속성 보강"
                score -= 0.03
            if name in existing_names:
                name += "후보"
                reason += "; Self-Correction: 기존 용어와 중복 보정"
                score -= 0.12
            results.setdefault(name, Recommendation(logical_name=name, description=description, reason=reason, score=max(0.0, score)))
        fallback_words = matched or words[:3]
        fallback_name = "".join(w.logical_name for w in fallback_words[:3]) or "표준용어"
        while len(results) < 3:
            index = len(results) + 1
            name = f"{fallback_name}후보{index}"
            reason = f"ToT 경로 {index}: " + ", ".join(w.logical_name for w in fallback_words[:3])
            reason += "; Self-Correction: 추천 후보 다양성 보강"
            results.setdefault(name, Recommendation(logical_name=name, description=description, reason=reason, score=max(0.0, 0.72 - index * 0.04)))
        return sorted(results.values(), key=lambda r: r.score, reverse=True)[:3]


class ValidateSkill:
    """신규/변경/삭제 요청을 표준 지침에 따라 검증하는 skill입니다."""

    def validate(self, request: ChangeRequest, repo: TermRepository) -> ValidationResult:
        """중복, 존재 여부, 설명 품질, 명명 규칙을 확인합니다."""

        current = repo.get_term(request.logical_name)
        if request.action == "delete":
            return ValidationResult(ok=current is not None, reason="변경이 가능합니다." if current else "존재하지 않는 표준용어라 삭제할 수 없습니다.", normalized_term=current)
        name = request.new_logical_name or request.logical_name
        desc = (request.description or "").strip()
        if len(name) < 3:
            return ValidationResult(ok=False, reason="용어 논리명은 3자 이상이어야 합니다.")
        if re.search(r"[^0-9A-Za-z가-힣_]", name):
            return ValidationResult(ok=False, reason="논리명에는 한글, 영문, 숫자, 밑줄만 사용할 수 있습니다.")
        if len(desc) < 8:
            return ValidationResult(ok=False, reason="설명은 8자 이상 작성해야 합니다.")
        if request.action == "insert" and repo.get_term(name):
            return ValidationResult(ok=False, reason=f"{name}은 이미 존재합니다.")
        if request.action == "update" and current is None:
            return ValidationResult(ok=False, reason=f"{request.logical_name} 표준용어가 존재하지 않습니다.")
        term = StandardTerm(logical_name=name, description=desc, physical_name=(current.physical_name if current else physical_name(name, repo.list_words())), domain=(current.domain if current else "common"))
        return ValidationResult(ok=True, reason="변경이 가능합니다.", normalized_term=term)


class DashGraph:
    """LangGraph 기반 Sub Agent 실행기입니다."""

    def __init__(self, tools: ToolBox) -> None:
        self.tools = tools
        self.recommend_skill = RecommendSkill()
        self.validate_skill = ValidateSkill()
        self.graph = self._build_graph()

    def invoke(self, state: AgentTurnState) -> AgentTurnState:
        """LangGraph 또는 fallback으로 한 턴을 실행합니다."""

        if self.graph is None:
            return self._fallback(state)
        return AgentTurnState.model_validate(self.graph.invoke(state.model_dump(), config={"configurable": {"thread_id": state.session_id}}))

    def _build_graph(self) -> Any:
        """Sub Agent 노드/엣지를 명확히 구성합니다."""

        if StateGraph is None:
            return None
        graph = StateGraph(dict)
        graph.add_node("select_agent", self._wrap(self.select_agent))
        graph.add_node("recommend_agent", self._wrap(self.recommend_agent))
        graph.add_node("validate_agent", self._wrap(self.validate_agent))
        graph.add_node("change_agent", self._wrap(self.change_agent))
        graph.add_node("finalize_agent", self._wrap(self.finalize_agent))
        graph.add_conditional_edges(START, self._route, {"select_agent": "select_agent", "recommend_agent": "recommend_agent", "validate_agent": "validate_agent", "finalize_agent": "finalize_agent"})
        graph.add_edge("select_agent", "finalize_agent")
        graph.add_edge("recommend_agent", "finalize_agent")
        graph.add_conditional_edges("validate_agent", self._after_validate, {"change_agent": "change_agent", "finalize_agent": "finalize_agent"})
        graph.add_edge("change_agent", "finalize_agent")
        graph.add_edge("finalize_agent", END)
        return graph.compile(checkpointer=MemorySaver() if MemorySaver else None)

    def _wrap(self, fn: Callable[[AgentTurnState], AgentTurnState]) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """LangGraph dict state와 Pydantic state를 변환합니다."""

        def node(raw: dict[str, Any]) -> dict[str, Any]:
            return fn(AgentTurnState.model_validate(raw)).model_dump()
        return node

    def _route(self, raw: dict[str, Any]) -> str:
        """Super Agent가 결정한 intent에 따라 다음 Sub Agent 노드를 선택합니다."""

        decision = AgentTurnState.model_validate(raw).router_decision
        if decision is None or decision.needs_clarification:
            return "finalize_agent"
        if decision.intent in {Intent.MEANING_SELECT, Intent.DA_REQUEST}:
            return "select_agent"
        if decision.intent == Intent.TERM_RECOMMEND:
            return "recommend_agent"
        return "validate_agent"

    def _after_validate(self, raw: dict[str, Any]) -> str:
        """검증 성공 시에만 변경 노드로 이동합니다."""

        validation = AgentTurnState.model_validate(raw).validation
        return "change_agent" if validation and validation.ok else "finalize_agent"

    def _fallback(self, state: AgentTurnState) -> AgentTurnState:
        """LangGraph 미설치 환경에서 동일한 노드 흐름을 실행합니다."""

        route = self._route(state.model_dump())
        if route == "select_agent":
            state = self.select_agent(state)
        elif route == "recommend_agent":
            state = self.recommend_agent(state)
        elif route == "validate_agent":
            state = self.validate_agent(state)
            if self._after_validate(state.model_dump()) == "change_agent":
                state = self.change_agent(state)
        return self.finalize_agent(state)

    def select_agent(self, state: AgentTurnState) -> AgentTurnState:
        """Vector DB, RDB, 웹 검색 tool을 호출합니다."""

        try:
            if state.router_decision and state.router_decision.intent == Intent.DA_REQUEST:
                state.final_answer = self.tools.tools["search_reference"].execute(state.user_input)
                state.logs.append(event("tool_execution_end", "select_agent", "문의 검색 완료"))
                return state
            if ambiguous(state.user_input):
                state.final_answer = "설명이 너무 짧거나 모호합니다. 예: '작업이 종료된 일시'처럼 입력해 주세요."
                return state
            state.similar_terms = self.tools.tools["search_similar_terms"].execute(state.user_input)
            state.lookup_count += 1
            state.final_answer = "\n".join(f"{x.term.logical_name}: {x.term.description} (similarity={x.similarity:.2f})" for x in state.similar_terms) or "유사도 0.6 이상인 기존 표준용어를 찾지 못했습니다. 추천을 요청할 수 있습니다."
            state.logs.append(event("tool_execution_end", "select_agent", "Vector DB 조회 완료", count=len(state.similar_terms), threshold=SIMILARITY_THRESHOLD))
        except Exception as exc:
            state.error = f"select_agent 오류: {exc}"
        return state

    def recommend_agent(self, state: AgentTurnState) -> AgentTurnState:
        """표준단어를 조회하고 ToT/Self-Correction으로 추천합니다."""

        try:
            description = state.user_input.replace("추천:", "").replace("추천", "").strip()
            if not description:
                for msg in reversed(state.memory):
                    if msg.details.get("intent") == Intent.MEANING_SELECT:
                        description = msg.content
                        break
            state.standard_words = self.tools.tools["list_standard_words"].execute()
            state.recommendations = self.recommend_skill.recommend(description, state.standard_words, self.tools.term_repository.list_terms())
            state.final_answer = "\n".join(x.logical_name for x in state.recommendations) or "추천 후보를 만들지 못했습니다."
            state.logs.append(event("tool_execution_end", "recommend_agent", "추천 완료", count=len(state.recommendations)))
        except Exception as exc:
            state.error = f"recommend_agent 오류: {exc}"
        return state

    def validate_agent(self, state: AgentTurnState) -> AgentTurnState:
        """insert/update/delete 전 검증을 수행합니다."""

        try:
            if state.router_decision is None:
                raise ValueError("router_decision 누락")
            e = state.router_decision.entities
            if state.router_decision.intent == Intent.TERM_DELETE:
                req = ChangeRequest(action="delete", logical_name=e.get("logical_name", state.user_input.strip("'\"` ")))
            elif state.router_decision.intent == Intent.TERM_UPDATE:
                req = ChangeRequest(action="update", logical_name=e.get("logical_name", ""), description=e.get("description"))
            else:
                req = ChangeRequest(action="insert", logical_name=e.get("logical_name", ""), description=e.get("description") or state.user_input)
            state.change_request = req
            state.validation = self.validate_skill.validate(req, self.tools.term_repository)
            state.logs.append(event("tool_execution_end", "validate_agent", state.validation.reason, ok=state.validation.ok, request=req.model_dump()))
        except Exception as exc:
            state.error = f"validate_agent 오류: {exc}"
        return state

    def change_agent(self, state: AgentTurnState) -> AgentTurnState:
        """검증된 요청을 RDB와 Vector DB에 반영합니다."""

        try:
            if not state.change_request or not state.validation or not state.validation.ok:
                raise ValueError("검증 성공 상태가 아닙니다.")
            state.final_answer = self.tools.tools["change_standard_term"].execute(state.change_request, state.validation.normalized_term)
            state.logs.append(event("tool_execution_end", "change_agent", state.final_answer))
        except Exception as exc:
            state.error = f"change_agent 오류: {exc}"
            state.final_answer = "변경 시 오류가 발생해 취소 처리되었습니다."
        return state

    def finalize_agent(self, state: AgentTurnState) -> AgentTurnState:
        """최종 응답을 생성합니다."""

        if state.error:
            state.final_answer = state.final_answer or state.error
        if state.router_decision and state.router_decision.needs_clarification:
            state.final_answer = state.router_decision.clarification_question or state.final_answer
        if state.validation and not state.validation.ok:
            state.final_answer = state.validation.reason
        state.final_answer = state.final_answer or "요청을 처리했습니다."
        state.messages.append(AgentMessage(role="assistant", content=state.final_answer))
        state.logs.append(event("agent_end", "finalize_agent", "최종 응답 생성", answer=state.final_answer))
        return state


class PiMonoSuperAgent:
    """Pi-mono 기반 Super Agent입니다."""

    def __init__(self, graph: DashGraph, llm: LlmBoundary) -> None:
        self.graph = graph
        self.llm = llm
        self.router_skill = RouterSkill()

    def route(self, state: AgentTurnState) -> RouterDecision:
        """사용자 입력을 분류하고 Sub Agent routing 결정을 반환합니다."""

        return self.router_skill.classify(state.user_input, state.lookup_count)

    def run(self, prompt: str, session_id: str, memory: list[AgentMessage], lookup_count: int) -> EventStream:
        """새 prompt를 context에 추가하고 Sub Agent graph를 실행합니다."""

        stream = EventStream()
        user_msg = AgentMessage(role="user", content=prompt)
        state = AgentTurnState(session_id=session_id, user_input=prompt, messages=[user_msg], memory=memory, lookup_count=lookup_count)
        for item in [event("agent_start", "super_agent", "시작"), event("turn_start", "super_agent", "턴 시작"), event("message_start", "user", prompt), event("message_end", "user", prompt)]:
            stream.push(item)
        try:
            state.router_decision = self.route(state)
            payload = state.router_decision.model_dump(mode="json")
            route_event = event("routing_end", "super_agent", "intent JSON 반환", json=payload)
            state.logs.append(route_event)
            state.messages.append(AgentMessage(role="toolResult", name="super_agent_router", content=json.dumps(payload, ensure_ascii=False)))
        except Exception as exc:
            state.error = f"super_agent routing 오류: {exc}"
        result = self.graph.invoke(state)
        for item in result.logs:
            stream.push(item)
        intent_value = result.router_decision.intent if result.router_decision else None
        result.memory = [*memory, AgentMessage(role="user", content=prompt, details={"intent": intent_value})]
        if result.messages and result.messages[-1].role == "assistant":
            result.memory.append(result.messages[-1])
        stream.push(event("turn_end", "super_agent", "턴 종료", lookup_count=result.lookup_count))
        stream.push(event("agent_end", "super_agent", "종료", answer=result.final_answer))
        stream.end(result)
        return stream


@dataclass
class Runtime:
    """앱 런타임 의존성 묶음입니다."""

    config: AppConfig
    repo: TermRepository
    vector: VectorRepository
    super_agent: PiMonoSuperAgent


def build_runtime() -> Runtime:
    """Mock 또는 운영 모드 런타임을 조립합니다."""

    config = AppConfig.from_env()
    if config.mock_mode:
        embedder: EmbeddingService = HashEmbeddingService()
        repo: TermRepository = MockTermRepository()
    else:
        if not config.oracle_url:
            raise RuntimeError("MOCK_MODE=false이면 ORACLE_URL이 필요합니다.")
        embedder = OpenAIEmbeddingService(config)
        repo = OracleTermRepository(config.oracle_url)
    vector: VectorRepository = MockVectorRepository(repo, embedder) if config.mock_mode else FaissVectorRepository(repo, embedder)
    tools = ToolBox(repo, vector, config)
    llm = LlmBoundary(config)
    return Runtime(config, repo, vector, PiMonoSuperAgent(DashGraph(tools), llm))


def init_ui() -> None:
    """Streamlit session memory와 turn 상태를 초기화합니다."""

    st.session_state.setdefault("session_id", str(uuid4()))
    st.session_state.setdefault("runtime", None)
    st.session_state.setdefault("memory", [])
    st.session_state.setdefault("lookup_count", 0)
    st.session_state.setdefault("answer", "")
    st.session_state.setdefault("lookup_answer", "")
    st.session_state.setdefault("similar", [])
    st.session_state.setdefault("recs", [])
    st.session_state.setdefault("selected_rec_idx", None)
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("last_description", "")
    st.session_state.setdefault("lookup_requested", False)


def css() -> None:
    """3분할 Streamlit 업무 화면 스타일을 적용합니다."""

    st.markdown("""
    <style>
    header[data-testid="stHeader"]{height:0;background:transparent;visibility:hidden}
    .stApp{background:#f6f7f8;color:#20262d;font-size:.92rem}
    .block-container{max-width:1760px;padding:.35rem .75rem 1rem}
    h1,h2,h3,.stMarkdown,.stText,.stTextArea,.stButton,.stCheckbox{font-size:.92rem}
    [data-testid="stMarkdownContainer"] p{font-size:.9rem;line-height:1.55}
    [data-testid="stTextArea"] textarea{font-size:.9rem;line-height:1.5}
    div.stButton>button{min-height:2.15rem;border-radius:6px;border:1px solid #337566;font-size:.88rem}
    .dash-title{font-family:Arial,Helvetica,sans-serif;font-size:2.75rem;font-weight:700;line-height:1;color:#17202a;margin:0 0 .35rem}
    .dash-lead{font-size:.92rem;line-height:1.6;color:#56616d;border-left:3px solid #337566;padding-left:.8rem;margin-bottom:1.1rem;max-width:980px}
    .panel-title{font-size:1rem;font-weight:700;margin:.35rem 0 .55rem;color:#20262d}
    .term-card{background:#ffffff;border:1px solid #d9dee3;border-radius:8px;padding:.75rem;margin-bottom:.55rem}
    .term-card b{font-size:.94rem}.term-card small{color:#66717d;font-size:.78rem}
    .rec-name{font-size:1.08rem;font-weight:700;color:#17202a;margin:.05rem 0 .3rem}
    .rec-reason{font-size:.76rem;line-height:1.45;color:#66717d}
    .rec-score{font-size:.72rem;color:#7b8792;margin-top:.25rem}
    .empty-card{background:#ffffff;border:1px dashed #c9d0d6;border-radius:8px;padding:.75rem;color:#66717d;font-size:.88rem}
    .result-box{background:#ffffff;border:1px solid #d9dee3;border-radius:8px;padding:.8rem;margin-bottom:.75rem}
    .agent-log{background:#17202a;color:#eef3f5;border-radius:8px;padding:.75rem;font-family:Consolas,monospace;font-size:.74rem;max-height:520px;overflow-y:auto;white-space:pre-wrap}
    </style>
    """, unsafe_allow_html=True)


def get_runtime() -> Runtime:
    """Streamlit rerun 사이에도 Mock RDB/Vector DB 상태를 유지합니다."""

    runtime = st.session_state.get("runtime")
    if runtime is None:
        runtime = build_runtime()
        st.session_state["runtime"] = runtime
    return cast(Runtime, runtime)


def execute_vector_lookup(runtime: Runtime, description: str) -> None:
    """조회 버튼 전용 Vector DB 검색을 수행합니다."""

    similar = runtime.vector.search(description)
    st.session_state["similar"] = similar
    st.session_state["lookup_count"] = cast(int, st.session_state["lookup_count"]) + 1
    st.session_state["lookup_requested"] = True
    st.session_state["last_description"] = description
    st.session_state["lookup_answer"] = (
        f"Vector DB 조회 완료: {len(similar)}건"
        if similar
        else "유사도 0.6 이상인 기존 표준용어를 찾지 못했습니다."
    )
    st.session_state["events"] = [
        f"event: tool_execution_end\ndata: Vector DB 조회 완료(count={len(similar)}, threshold={SIMILARITY_THRESHOLD})"
    ]


def execute_prompt(runtime: Runtime, prompt: str) -> AgentTurnState:
    """UI 입력을 Super Agent에 전달하고 session_state를 갱신합니다."""

    stream = runtime.super_agent.run(prompt, st.session_state["session_id"], cast(list[AgentMessage], st.session_state["memory"]), cast(int, st.session_state["lookup_count"]))
    result = cast(AgentTurnState, stream.result)
    st.session_state["memory"] = result.memory
    st.session_state["lookup_count"] = result.lookup_count
    st.session_state["answer"] = result.final_answer
    st.session_state["similar"] = result.similar_terms
    st.session_state["recs"] = result.recommendations
    st.session_state["events"] = stream.sse()
    if result.router_decision and result.router_decision.intent == Intent.MEANING_SELECT and not result.router_decision.needs_clarification:
        st.session_state["last_description"] = prompt
    return result


def render_similar_terms() -> None:
    """Vector DB 조회결과를 카드로 표시합니다."""

    st.markdown('<div class="panel-title">Vector DB 조회결과</div>', unsafe_allow_html=True)
    if st.session_state["lookup_answer"]:
        st.markdown(f'<div class="result-box">{st.session_state["lookup_answer"]}</div>', unsafe_allow_html=True)
    similar_terms = cast(list[SimilarTerm], st.session_state["similar"])
    if not similar_terms:
        st.markdown('<div class="empty-card">조회 버튼을 누르면 이 영역에 결과가 표시됩니다.</div>', unsafe_allow_html=True)
        return
    for item in similar_terms:
        st.markdown(
            f'<div class="term-card"><b>{item.term.logical_name}</b><br>{item.term.description}<br><small>similarity={item.similarity:.2f}</small></div>',
            unsafe_allow_html=True,
        )


def render_recommendations(runtime: Runtime) -> None:
    """추천 용어를 체크박스와 등록 버튼으로 표시합니다."""

    st.markdown('<div class="panel-title">추천 결과</div>', unsafe_allow_html=True)
    recommendations = cast(list[Recommendation], st.session_state["recs"])
    if not recommendations:
        st.markdown('<div class="empty-card">추천 버튼을 누르면 표준단어 조합 후보가 표시됩니다.</div>', unsafe_allow_html=True)
        return

    selected_idx = cast(int | None, st.session_state["selected_rec_idx"])
    for i, rec in enumerate(recommendations):
        check_col, content_col = st.columns([0.13, 0.87], gap="small")
        with check_col:
            checked = st.checkbox(
                rec.logical_name,
                value=selected_idx == i,
                key=f"rec_checkbox_{i}",
                help=rec.reason,
                label_visibility="collapsed",
            )
        if checked and selected_idx != i:
            st.session_state["selected_rec_idx"] = i
            selected_idx = i
        elif not checked and selected_idx == i:
            st.session_state["selected_rec_idx"] = None
            selected_idx = None
        with content_col:
            st.markdown(
                f'<div class="term-card"><div class="rec-name">{rec.logical_name}</div><div class="rec-reason">{rec.reason}</div><div class="rec-score">score={rec.score:.2f}</div></div>',
                unsafe_allow_html=True,
            )

    if st.button("등록", disabled=selected_idx is None, use_container_width=True):
        rec = recommendations[cast(int, selected_idx)]
        result = execute_prompt(runtime, f"'{rec.logical_name}' 등록: {rec.description}")
        st.session_state["answer"] = result.final_answer


def render_change_result() -> None:
    """등록/변경 결과를 변경 버튼 하단 영역에 표시합니다."""

    st.markdown('<div class="panel-title">등록/변경 결과</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-box">{st.session_state["answer"] or "아직 등록 또는 변경 결과가 없습니다."}</div>',
        unsafe_allow_html=True,
    )


def render_app() -> None:
    """Streamlit 화면을 렌더링합니다."""

    if st is None:
        raise RuntimeError("Streamlit이 설치되어 있지 않습니다.")
    st.set_page_config(page_title="DAsh", page_icon="DA", layout="wide")
    init_ui()
    css()
    runtime = get_runtime()
    st.markdown('<div class="dash-title">DAsh</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-lead">DAsh는 표준용어 관리를 돕는 Pi-mono Super Agent입니다. 사용자 설명을 Vector DB에서 조회하고, 표준단어를 조합해 용어 후보를 추천하며, 검증된 등록/수정/삭제 요청을 RDB와 Vector DB에 순차 반영합니다.</div>', unsafe_allow_html=True)

    work_area, log_area = st.columns([3.8, 1], gap="large")
    with work_area:
        st.markdown('<div class="panel-title">사용자 입력</div>', unsafe_allow_html=True)
        user_text = st.text_area(
            "입력",
            height=150,
            placeholder="작업이 시작된 일시\n'작업시작일시'의 설명을 '작업이 시작된 일시'로 수정\n'작업시작일시' 삭제",
            label_visibility="collapsed",
        )
        lookup_col, recommend_col, change_col = st.columns([1, 1, 1], gap="small")
        with lookup_col:
            if st.button("조회", use_container_width=True) and user_text.strip():
                execute_vector_lookup(runtime, user_text.strip())
        with recommend_col:
            recommend_disabled = not cast(bool, st.session_state["lookup_requested"])
            if st.button("추천", disabled=recommend_disabled, use_container_width=True):
                prompt = "추천: " + (user_text.strip() or cast(str, st.session_state["last_description"]))
                result = execute_prompt(runtime, prompt)
                st.session_state["selected_rec_idx"] = None
                st.session_state["answer"] = result.final_answer
        with change_col:
            if st.button("변경", use_container_width=True) and user_text.strip():
                result = execute_prompt(runtime, user_text.strip())
                st.session_state["answer"] = result.final_answer

        lookup_result_col, recommend_result_col, change_result_col = st.columns([1, 1, 1], gap="small")
        with lookup_result_col:
            render_similar_terms()
        with recommend_result_col:
            render_recommendations(runtime)
        with change_result_col:
            render_change_result()

    with log_area:
        st.markdown('<div class="panel-title">Agent 처리 로그</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="agent-log">{"<br><br>".join(cast(list[str], st.session_state["events"])) or "SSE 이벤트가 여기에 표시됩니다."}</div>',
            unsafe_allow_html=True,
        )


def streamlit_context() -> bool:
    """streamlit run으로 실행 중인지 확인합니다."""

    if st is None:
        return False
    try:
        import logging
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").disabled = True
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_cli(args: list[str]) -> int:
    """CLI에서 Mock 시나리오를 실행합니다."""

    prompt = " ".join(args).strip() or "작업이 시작된 일시"
    runtime = build_runtime()
    stream = runtime.super_agent.run(prompt, str(uuid4()), [], 0)
    result = cast(AgentTurnState, stream.result)
    print(result.final_answer)
    print("\n--- SSE ---")
    for line in stream.sse():
        print(line)
    print("\n실행 방법: streamlit run 05.09_prompt_v5.py")
    return 0


if __name__ == "__main__":
    if streamlit_context():
        render_app()
    else:
        raise SystemExit(run_cli(sys.argv[1:]))
