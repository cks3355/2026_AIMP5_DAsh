from __future__ import annotations

"""
표준용어 Multi Agent 시스템 예제입니다.

이 파일은 Streamlit UI, LangGraph Agent 흐름, Mock RDB, FAISS Vector DB 교체 지점을
하나의 실행 가능한 Python 파일에 모은 구현입니다. 실제 Oracle/FAISS/OpenAI 환경이 준비되지
않아도 Mock 데이터와 hashing embedding으로 주요 시나리오를 검증할 수 있습니다.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Sequence, TypedDict, cast
from uuid import uuid4

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 선택 의존성 로딩 영역입니다.
# 개발/운영 환경에 따라 LangGraph, Streamlit, FAISS, Oracle client가 없을 수 있으므로
# import 실패 시 명확한 fallback 또는 오류 메시지를 제공하도록 분리합니다.
# -----------------------------------------------------------------------------
try:
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit 미설치 CLI 환경 보호
    st = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - python-dotenv 미설치 환경 보호

    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False


try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except Exception:  # pragma: no cover - 최소 CLI 안내를 위한 fallback

    class BaseMessage:  # type: ignore[no-redef]
        def __init__(self, content: str) -> None:
            self.content = content

    class HumanMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    class AIMessage(BaseMessage):  # type: ignore[no-redef]
        pass

    class SystemMessage(BaseMessage):  # type: ignore[no-redef]
        pass


LANGGRAPH_IMPORT_ERROR: Exception | None = None
try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
except Exception as exc:  # pragma: no cover - 의존성 안내용
    LANGGRAPH_IMPORT_ERROR = exc
    MemorySaver = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]

    def add_messages(left: list[BaseMessage] | None, right: list[BaseMessage] | None) -> list[BaseMessage]:
        return [*(left or []), *(right or [])]


try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - Mock embedding fallback 사용
    ChatOpenAI = None  # type: ignore[assignment]
    OpenAIEmbeddings = None  # type: ignore[assignment]

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except Exception:  # pragma: no cover - Tavily 미설치/미설정 환경 보호
    TavilySearchResults = None  # type: ignore[assignment]

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine, URL
except Exception:  # pragma: no cover - Mock RDB fallback 사용
    create_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]
    Engine = Any  # type: ignore[misc,assignment]
    URL = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# OpenTelemetry는 Agent 노드 단위의 추적 span을 남기기 위해 사용합니다.
# OTLP endpoint가 있으면 collector로 보내고, 없으면 no-op 또는 선택적 console exporter로 동작합니다.
# 내부 Chain-of-Thought를 저장하지 않고, 입력/출력 요약, 라우팅 판단 근거, tool 결과만 기록합니다.
# -----------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except Exception:  # pragma: no cover
        OTLPSpanExporter = None  # type: ignore[assignment]
except Exception:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    ConsoleSpanExporter = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]


class NoOpSpan:
    """OpenTelemetry가 없는 환경에서도 동일한 문법으로 span을 사용할 수 있게 하는 no-op span입니다."""

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def set_attribute(self, *_: Any, **__: Any) -> None:
        return None

    def record_exception(self, *_: Any, **__: Any) -> None:
        return None


class NoOpTracer:
    """OpenTelemetry tracer fallback입니다."""

    def start_as_current_span(self, *_: Any, **__: Any) -> NoOpSpan:
        return NoOpSpan()


# -----------------------------------------------------------------------------
# 전역 상수입니다.
# 요구사항에 따라 유사도 top-k와 threshold는 코드 기준값으로 고정합니다.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SIMILARITY_TOP_K = 3
SIMILARITY_THRESHOLD = 0.6
DEFAULT_DOMAIN = "공통"
SUPPORTED_INTENTS = {
    "meaning_search",
    "term_recommend",
    "term_register",
    "term_change",
    "request",
}

IntentName = Literal["meaning_search", "term_recommend", "term_register", "term_change", "request"]
ChangeAction = Literal["update", "delete"]


# -----------------------------------------------------------------------------
# 로그 설정입니다.
# Streamlit 화면에는 state.logs를 보여주고, 서버 콘솔에는 Python logging을 남깁니다.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("standard-term-agent")


def append_list(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """LangGraph state에서 로그 목록을 누적하기 위한 reducer입니다."""

    return [*(left or []), *(right or [])]


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Pydantic v1/v2 양쪽에서 모델을 dict로 변환합니다."""

    if hasattr(model, "model_dump"):
        return cast(dict[str, Any], model.model_dump())
    return cast(dict[str, Any], model.dict())


def now_iso() -> str:
    """로그 타임스탬프를 UTC ISO-8601 문자열로 반환합니다."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_log(agent: str, event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Streamlit에서 확인할 수 있는 Agent 처리 로그 항목을 생성합니다."""

    return {
        "timestamp": now_iso(),
        "agent": agent,
        "event": event,
        "payload": payload or {},
    }


def to_json(data: Any) -> str:
    """의도 분류 결과와 로그를 한글이 깨지지 않는 JSON 문자열로 직렬화합니다."""

    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# -----------------------------------------------------------------------------
# .env 기반 설정 모델입니다.
# DB 접속 정보, OpenAI/LangSmith/Tavily/OpenTelemetry 설정은 모두 여기에서 읽습니다.
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    """애플리케이션 실행 설정입니다."""

    use_mock_data: bool
    llm_model: str
    embedding_model: str
    openai_api_key: str | None
    openai_base_url: str | None
    embedding_base_url: str | None
    oracle_database_url: str | None
    oracle_user: str | None
    oracle_password: str | None
    oracle_host: str | None
    oracle_port: int
    oracle_service_name: str | None
    tavily_api_key: str | None
    langsmith_api_key: str | None
    langsmith_project: str
    otel_endpoint: str | None
    otel_console: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        """현재 디렉터리의 .env를 읽어 실행 설정을 생성합니다."""

        load_dotenv(BASE_DIR / ".env", override=True)
        openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        oracle_database_url = os.getenv("ORACLE_DATABASE_URL")
        use_mock_data = env_bool("USE_MOCK_DATA", default=not bool(oracle_database_url))
        return cls(
            use_mock_data=use_mock_data,
            llm_model=os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-5",
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            openai_api_key=openai_api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL"),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
            oracle_database_url=oracle_database_url,
            oracle_user=os.getenv("ORACLE_USER"),
            oracle_password=os.getenv("ORACLE_PASSWORD"),
            oracle_host=os.getenv("ORACLE_HOST"),
            oracle_port=int(os.getenv("ORACLE_PORT", "1521")),
            oracle_service_name=os.getenv("ORACLE_SERVICE_NAME"),
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "standard-term-agent"),
            otel_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otel_console=env_bool("OTEL_CONSOLE_EXPORTER", default=False),
        )


def env_bool(name: str, default: bool) -> bool:
    """문자열 환경변수를 bool 값으로 변환합니다."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def configure_langsmith(config: AppConfig) -> None:
    """LangSmith tracing을 활성화합니다."""

    if not config.langsmith_api_key:
        return
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", config.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", config.langsmith_project)


def configure_opentelemetry(config: AppConfig) -> Any:
    """OpenTelemetry tracer를 구성하고 반환합니다."""

    if trace is None or TracerProvider is None or BatchSpanProcessor is None:
        return NoOpTracer()

    with contextlib.suppress(Exception):
        provider = TracerProvider()
        if config.otel_endpoint and OTLPSpanExporter is not None:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otel_endpoint)))
        elif config.otel_console and ConsoleSpanExporter is not None:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    return trace.get_tracer("standard-term-agent")


# -----------------------------------------------------------------------------
# 상태 모델입니다.
# messages는 LangGraph add_messages reducer를 사용해 대화 턴을 누적합니다.
# logs는 append_list reducer를 사용해 각 Agent의 처리 이벤트를 누적합니다.
# -----------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    logs: Annotated[list[dict[str, Any]], append_list]
    user_input: str
    forced_intent: IntentName
    intent: IntentName
    classification_json: str
    route_reason: str
    response: str
    search_count: int
    last_description: str
    last_search_results: list[dict[str, Any]]
    term_results: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    selected_term: str
    active_description: str
    validation: dict[str, Any]
    pending_change_request: dict[str, Any]
    apply_change: bool
    change_result: dict[str, Any]
    turn_id: str


# -----------------------------------------------------------------------------
# 업무 데이터 모델입니다.
# Oracle 스키마와 Mock 스키마를 같은 형태로 맞추기 위해 Pydantic 모델을 사용합니다.
# -----------------------------------------------------------------------------
class StandardWord(BaseModel):
    """표준단어입니다."""

    logical_name: str
    physical_name: str = ""
    word_type: Literal["business", "class"] = "business"
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)


class StandardTerm(BaseModel):
    """표준용어입니다."""

    logical_name: str
    description: str
    domain: str = DEFAULT_DOMAIN
    status: str = "ACTIVE"


class SimilarTermResult(BaseModel):
    """Vector DB 검색 결과입니다."""

    logical_name: str
    description: str
    domain: str = DEFAULT_DOMAIN
    similarity: float


class IntentDecision(BaseModel):
    """router_agent가 반드시 JSON으로 반환해야 하는 의도 분류 결과입니다."""

    intent: IntentName
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str
    normalized_input: str


class TermProposal(BaseModel):
    """recommend_agent가 생성하는 신규 표준용어 후보입니다."""

    logical_name: str
    used_words: list[str]
    score: float
    reason: str


class ValidationReport(BaseModel):
    """validate_agent의 검증 결과입니다."""

    ok: bool
    reasons: list[str] = Field(default_factory=list)
    target_intent: IntentName
    logical_name: str | None = None
    description: str | None = None


class ChangePlan(BaseModel):
    """변경 요청을 실제 update/delete 작업으로 정규화한 계획입니다."""

    action: ChangeAction
    logical_name: str
    new_description: str | None = None
    requested_text: str


# -----------------------------------------------------------------------------
# RDB 접근 추상화입니다.
# MockStandardRepository는 즉시 실행 가능한 테스트용이고, OracleStandardRepository는 SQLAlchemy로 교체됩니다.
# -----------------------------------------------------------------------------
class StandardRepository(Protocol):
    """표준단어/표준용어 저장소 인터페이스입니다."""

    def list_words(self) -> list[StandardWord]:
        """활성 표준단어 목록을 반환합니다."""

    def list_terms(self) -> list[StandardTerm]:
        """활성 표준용어 목록을 반환합니다."""

    def get_term(self, logical_name: str) -> StandardTerm | None:
        """논리명으로 표준용어를 조회합니다."""

    def search_terms(self, keyword: str, limit: int = 5) -> list[StandardTerm]:
        """RDB에서 키워드 기반 표준용어를 조회합니다."""

    def find_words(self, keyword: str) -> list[StandardWord]:
        """RDB에서 키워드 기반 표준단어를 조회합니다."""

    def insert_term(self, term: StandardTerm) -> None:
        """표준용어를 신규 등록합니다."""

    def update_term_description(self, logical_name: str, description: str) -> None:
        """표준용어 설명을 수정합니다."""

    def delete_term(self, logical_name: str) -> None:
        """표준용어를 삭제 또는 비활성화합니다."""


class MockStandardRepository:
    """Oracle 없이도 실행 가능한 in-memory RDB mock입니다."""

    def __init__(self) -> None:
        self._words: dict[str, StandardWord] = {
            "작업": StandardWord(
                logical_name="작업",
                physical_name="JOB",
                word_type="business",
                description="업무 또는 공정에서 수행되는 단위 활동입니다.",
                synonyms=["업무", "task", "job", "work"],
            ),
            "시작": StandardWord(
                logical_name="시작",
                physical_name="STRT",
                word_type="business",
                description="행위나 상태가 처음 발생하는 것을 의미합니다.",
                synonyms=["개시", "start", "착수"],
            ),
            "종료": StandardWord(
                logical_name="종료",
                physical_name="END",
                word_type="business",
                description="행위나 상태가 끝나는 것을 의미합니다.",
                synonyms=["완료", "마감", "end", "finish"],
            ),
            "일시": StandardWord(
                logical_name="일시",
                physical_name="DTTM",
                word_type="class",
                description="일자와 시각을 함께 표현하는 분류어입니다.",
                synonyms=["시점", "시간", "일자시간", "timestamp", "datetime"],
            ),
            "재공": StandardWord(
                logical_name="재공",
                physical_name="WIP",
                word_type="business",
                description="제조 공정에서 생산 중이거나 가공 중인 미완성 제품입니다.",
                synonyms=["공정중", "미완성품", "work in process", "wip"],
            ),
            "수량": StandardWord(
                logical_name="수량",
                physical_name="QTY",
                word_type="class",
                description="개수나 양을 나타내는 분류어입니다.",
                synonyms=["개수", "양", "quantity", "qty"],
            ),
            "고객": StandardWord(
                logical_name="고객",
                physical_name="CUST",
                word_type="business",
                description="상품 또는 서비스를 이용하는 개인이나 법인입니다.",
                synonyms=["손님", "거래처", "customer"],
            ),
            "ID": StandardWord(
                logical_name="ID",
                physical_name="ID",
                word_type="class",
                description="대상을 고유하게 식별하는 값입니다.",
                synonyms=["아이디", "식별자", "identifier"],
            ),
            "계좌": StandardWord(
                logical_name="계좌",
                physical_name="ACCT",
                word_type="business",
                description="금융 거래를 위해 개설된 관리 단위입니다.",
                synonyms=["account", "계정"],
            ),
            "번호": StandardWord(
                logical_name="번호",
                physical_name="NO",
                word_type="class",
                description="대상을 구분하기 위해 부여하는 숫자 또는 문자열입니다.",
                synonyms=["no", "number"],
            ),
            "거래": StandardWord(
                logical_name="거래",
                physical_name="TRX",
                word_type="business",
                description="상품, 서비스, 금전 등이 오가는 행위입니다.",
                synonyms=["매매", "transaction"],
            ),
            "일자": StandardWord(
                logical_name="일자",
                physical_name="DT",
                word_type="class",
                description="연월일 단위의 날짜를 의미합니다.",
                synonyms=["날짜", "date"],
            ),
            "상품": StandardWord(
                logical_name="상품",
                physical_name="PRD",
                word_type="business",
                description="판매 또는 관리 대상이 되는 재화나 서비스입니다.",
                synonyms=["제품", "product"],
            ),
            "코드": StandardWord(
                logical_name="코드",
                physical_name="CD",
                word_type="class",
                description="대상을 체계적으로 구분하기 위한 기호 값입니다.",
                synonyms=["code", "분류값"],
            ),
        }
        self._terms: dict[str, StandardTerm] = {
            "작업시작일시": StandardTerm(
                logical_name="작업시작일시",
                description="작업이 start된 시점",
                domain="생산",
            ),
            "재공수량": StandardTerm(
                logical_name="재공수량",
                description="제조 공정에서 생산 중이거나 가공 중인 미완성 제품의 수량",
                domain="제조",
            ),
            "고객ID": StandardTerm(
                logical_name="고객ID",
                description="고객을 고유하게 식별하기 위해 부여한 ID",
                domain="고객",
            ),
            "계좌번호": StandardTerm(
                logical_name="계좌번호",
                description="금융 거래 계좌를 구분하기 위해 부여한 번호",
                domain="금융",
            ),
            "거래일자": StandardTerm(
                logical_name="거래일자",
                description="거래가 실제로 발생한 날짜",
                domain="금융",
            ),
            "상품코드": StandardTerm(
                logical_name="상품코드",
                description="상품이나 서비스를 식별하기 위해 정의한 코드",
                domain="상품",
            ),
        }

    def list_words(self) -> list[StandardWord]:
        return list(self._words.values())

    def list_terms(self) -> list[StandardTerm]:
        return [term for term in self._terms.values() if term.status == "ACTIVE"]

    def get_term(self, logical_name: str) -> StandardTerm | None:
        term = self._terms.get(logical_name)
        if term and term.status == "ACTIVE":
            return term
        return None

    def search_terms(self, keyword: str, limit: int = 5) -> list[StandardTerm]:
        compact = normalize_text(keyword)
        results: list[StandardTerm] = []
        for term in self.list_terms():
            haystack = normalize_text(f"{term.logical_name} {term.description} {term.domain}")
            if compact and compact in haystack:
                results.append(term)
            elif any(token and token in haystack for token in tokenize(keyword)):
                results.append(term)
            if len(results) >= limit:
                break
        return results

    def find_words(self, keyword: str) -> list[StandardWord]:
        compact = normalize_text(keyword)
        matches: list[StandardWord] = []
        for word in self.list_words():
            aliases = [word.logical_name, word.physical_name, *word.synonyms]
            if any(normalize_text(alias) and normalize_text(alias) in compact for alias in aliases):
                matches.append(word)
        return matches

    def insert_term(self, term: StandardTerm) -> None:
        if term.logical_name in self._terms and self._terms[term.logical_name].status == "ACTIVE":
            raise ValueError(f"이미 등록된 표준용어입니다: {term.logical_name}")
        self._terms[term.logical_name] = term

    def update_term_description(self, logical_name: str, description: str) -> None:
        current = self.get_term(logical_name)
        if current is None:
            raise ValueError(f"표준용어를 찾을 수 없습니다: {logical_name}")
        self._terms[logical_name] = current.copy(update={"description": description})

    def delete_term(self, logical_name: str) -> None:
        current = self.get_term(logical_name)
        if current is None:
            raise ValueError(f"표준용어를 찾을 수 없습니다: {logical_name}")
        self._terms[logical_name] = current.copy(update={"status": "DELETED"})


class OracleStandardRepository:
    """SQLAlchemy 기반 Oracle RDB 저장소입니다."""

    def __init__(self, config: AppConfig) -> None:
        if create_engine is None or text is None:
            raise RuntimeError("SQLAlchemy가 설치되어 있지 않아 Oracle RDB를 사용할 수 없습니다.")
        self.engine = create_engine(self._build_oracle_url(config), pool_pre_ping=True)

    def _build_oracle_url(self, config: AppConfig) -> str | Any:
        """ORACLE_DATABASE_URL 또는 ORACLE_* 조합으로 SQLAlchemy URL을 만듭니다."""

        if config.oracle_database_url:
            return config.oracle_database_url
        if not (config.oracle_user and config.oracle_password and config.oracle_host and config.oracle_service_name):
            raise RuntimeError("Oracle 연결 정보가 부족합니다. .env에 ORACLE_DATABASE_URL 또는 ORACLE_* 값을 설정하세요.")
        if URL is None:
            raise RuntimeError("SQLAlchemy URL helper를 사용할 수 없습니다.")
        return URL.create(
            "oracle+oracledb",
            username=config.oracle_user,
            password=config.oracle_password,
            host=config.oracle_host,
            port=config.oracle_port,
            query={"service_name": config.oracle_service_name},
        )

    def list_words(self) -> list[StandardWord]:
        query = text(
            """
            SELECT logical_name, physical_name, word_type, description, synonyms
            FROM standard_words
            WHERE status = 'ACTIVE'
            ORDER BY logical_name
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [self._row_to_word(row) for row in rows]

    def list_terms(self) -> list[StandardTerm]:
        query = text(
            """
            SELECT logical_name, description, domain, status
            FROM standard_terms
            WHERE status = 'ACTIVE'
            ORDER BY logical_name
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
        return [self._row_to_term(row) for row in rows]

    def get_term(self, logical_name: str) -> StandardTerm | None:
        query = text(
            """
            SELECT logical_name, description, domain, status
            FROM standard_terms
            WHERE logical_name = :logical_name AND status = 'ACTIVE'
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(query, {"logical_name": logical_name}).mappings().first()
        return self._row_to_term(row) if row else None

    def search_terms(self, keyword: str, limit: int = 5) -> list[StandardTerm]:
        query = text(
            """
            SELECT logical_name, description, domain, status
            FROM standard_terms
            WHERE status = 'ACTIVE'
              AND (logical_name LIKE :keyword OR description LIKE :keyword)
            FETCH FIRST :limit ROWS ONLY
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"keyword": f"%{keyword}%", "limit": limit}).mappings().all()
        return [self._row_to_term(row) for row in rows]

    def find_words(self, keyword: str) -> list[StandardWord]:
        query = text(
            """
            SELECT logical_name, physical_name, word_type, description, synonyms
            FROM standard_words
            WHERE status = 'ACTIVE'
              AND (logical_name LIKE :keyword OR synonyms LIKE :keyword OR description LIKE :keyword)
            FETCH FIRST 10 ROWS ONLY
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(query, {"keyword": f"%{keyword}%"}).mappings().all()
        return [self._row_to_word(row) for row in rows]

    def insert_term(self, term_value: StandardTerm) -> None:
        query = text(
            """
            INSERT INTO standard_terms(logical_name, description, domain, status, created_at, updated_at)
            VALUES (:logical_name, :description, :domain, 'ACTIVE', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
        )
        with self.engine.begin() as conn:
            conn.execute(
                query,
                {
                    "logical_name": term_value.logical_name,
                    "description": term_value.description,
                    "domain": term_value.domain,
                },
            )

    def update_term_description(self, logical_name: str, description: str) -> None:
        query = text(
            """
            UPDATE standard_terms
            SET description = :description, updated_at = CURRENT_TIMESTAMP
            WHERE logical_name = :logical_name AND status = 'ACTIVE'
            """
        )
        with self.engine.begin() as conn:
            result = conn.execute(query, {"logical_name": logical_name, "description": description})
            if result.rowcount == 0:
                raise ValueError(f"수정 대상 표준용어가 없습니다: {logical_name}")

    def delete_term(self, logical_name: str) -> None:
        query = text(
            """
            UPDATE standard_terms
            SET status = 'DELETED', updated_at = CURRENT_TIMESTAMP
            WHERE logical_name = :logical_name AND status = 'ACTIVE'
            """
        )
        with self.engine.begin() as conn:
            result = conn.execute(query, {"logical_name": logical_name})
            if result.rowcount == 0:
                raise ValueError(f"삭제 대상 표준용어가 없습니다: {logical_name}")

    def _row_to_word(self, row: Any) -> StandardWord:
        synonyms_raw = str(row.get("synonyms") or "")
        synonyms = [item.strip() for item in re.split(r"[,|/]", synonyms_raw) if item.strip()]
        word_type = str(row.get("word_type") or "business").lower()
        return StandardWord(
            logical_name=str(row.get("logical_name") or ""),
            physical_name=str(row.get("physical_name") or ""),
            word_type="class" if word_type == "class" else "business",
            description=str(row.get("description") or ""),
            synonyms=synonyms,
        )

    def _row_to_term(self, row: Any) -> StandardTerm:
        return StandardTerm(
            logical_name=str(row.get("logical_name") or ""),
            description=str(row.get("description") or ""),
            domain=str(row.get("domain") or DEFAULT_DOMAIN),
            status=str(row.get("status") or "ACTIVE"),
        )


# -----------------------------------------------------------------------------
# Embedding 및 Vector DB 계층입니다.
# OpenAI embedding이 설정되어 있으면 text-embedding-3-large를 사용하고, 아니면 Mock 실행을 위해
# 표준단어 동의어를 반영한 deterministic hashing embedding을 사용합니다.
# -----------------------------------------------------------------------------
class EmbeddingProvider(Protocol):
    """Embedding provider 인터페이스입니다."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """문서 목록을 벡터로 변환합니다."""

    def embed_query(self, text_value: str) -> list[float]:
        """검색 질의를 벡터로 변환합니다."""


class LangChainOpenAIEmbeddingProvider:
    """LangChain OpenAI embedding wrapper입니다."""

    def __init__(self, config: AppConfig) -> None:
        if OpenAIEmbeddings is None:
            raise RuntimeError("langchain-openai가 설치되어 있지 않습니다.")
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY 또는 LLM_API_KEY가 설정되어 있지 않습니다.")

        kwargs: dict[str, Any] = {
            "model": config.embedding_model,
            "api_key": config.openai_api_key,
        }
        if config.embedding_base_url:
            kwargs["base_url"] = config.embedding_base_url
        elif config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url
        self._embeddings = OpenAIEmbeddings(**kwargs)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return cast(list[list[float]], self._embeddings.embed_documents(list(texts)))

    def embed_query(self, text_value: str) -> list[float]:
        return cast(list[float], self._embeddings.embed_query(text_value))


class HashingEmbeddingProvider:
    """Mock 실행을 위한 deterministic hashing embedding입니다."""

    def __init__(self, words: Sequence[StandardWord], dimensions: int = 384) -> None:
        self.words = list(words)
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text_value) for text_value in texts]

    def embed_query(self, text_value: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = expand_tokens_with_standard_words(text_value, self.words)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
            index = int(digest, 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class TermVectorStore(Protocol):
    """표준용어 Vector DB 인터페이스입니다."""

    def search(self, query: str, top_k: int = SIMILARITY_TOP_K) -> list[SimilarTermResult]:
        """유사도 상위 표준용어를 반환합니다."""

    def upsert_term(self, term_value: StandardTerm) -> None:
        """표준용어 벡터를 추가 또는 갱신합니다."""

    def delete_term(self, logical_name: str) -> None:
        """표준용어 벡터를 삭제합니다."""

    def rebuild(self, terms: Sequence[StandardTerm]) -> None:
        """RDB 기준으로 Vector DB를 재생성합니다."""


class InMemoryTermVectorStore:
    """FAISS가 없을 때 사용하는 cosine search fallback입니다."""

    def __init__(self, embedding_provider: EmbeddingProvider, terms: Sequence[StandardTerm]) -> None:
        self.embedding_provider = embedding_provider
        self.terms: list[StandardTerm] = []
        self.vectors: list[list[float]] = []
        self.rebuild(terms)

    def rebuild(self, terms: Sequence[StandardTerm]) -> None:
        self.terms = list(terms)
        self.vectors = self.embedding_provider.embed_texts([term_to_vector_text(term) for term in self.terms])

    def search(self, query: str, top_k: int = SIMILARITY_TOP_K) -> list[SimilarTermResult]:
        query_vector = self.embedding_provider.embed_query(query)
        scored: list[SimilarTermResult] = []
        for term, vector in zip(self.terms, self.vectors):
            scored.append(
                SimilarTermResult(
                    logical_name=term.logical_name,
                    description=term.description,
                    domain=term.domain,
                    similarity=round(cosine_similarity(query_vector, vector), 4),
                )
            )
        return sorted(scored, key=lambda item: item.similarity, reverse=True)[:top_k]

    def upsert_term(self, term_value: StandardTerm) -> None:
        self.terms = [term for term in self.terms if term.logical_name != term_value.logical_name]
        self.terms.append(term_value)
        self.rebuild(self.terms)

    def delete_term(self, logical_name: str) -> None:
        self.terms = [term for term in self.terms if term.logical_name != logical_name]
        self.rebuild(self.terms)


class FaissTermVectorStore:
    """FAISS IndexFlatIP 기반 Vector DB 구현입니다."""

    def __init__(self, embedding_provider: EmbeddingProvider, terms: Sequence[StandardTerm]) -> None:
        self.embedding_provider = embedding_provider
        self.terms: list[StandardTerm] = []
        self.index: Any | None = None
        self._faiss: Any | None = None
        self._np: Any | None = None
        self.rebuild(terms)

    def _load_faiss(self) -> tuple[Any, Any]:
        """faiss와 numpy를 동적으로 로딩합니다."""

        if self._faiss is None or self._np is None:
            import faiss  # type: ignore
            import numpy as np  # type: ignore

            self._faiss = faiss
            self._np = np
        return self._faiss, self._np

    def rebuild(self, terms: Sequence[StandardTerm]) -> None:
        self.terms = list(terms)
        faiss, np = self._load_faiss()
        if not self.terms:
            self.index = None
            return
        vectors = self.embedding_provider.embed_texts([term_to_vector_text(term) for term in self.terms])
        matrix = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def search(self, query: str, top_k: int = SIMILARITY_TOP_K) -> list[SimilarTermResult]:
        if self.index is None or not self.terms:
            return []
        faiss, np = self._load_faiss()
        query_vector = np.asarray([self.embedding_provider.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vector)
        scores, indexes = self.index.search(query_vector, min(top_k, len(self.terms)))
        results: list[SimilarTermResult] = []
        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue
            term = self.terms[int(index)]
            results.append(
                SimilarTermResult(
                    logical_name=term.logical_name,
                    description=term.description,
                    domain=term.domain,
                    similarity=round(float(score), 4),
                )
            )
        return results

    def upsert_term(self, term_value: StandardTerm) -> None:
        self.terms = [term for term in self.terms if term.logical_name != term_value.logical_name]
        self.terms.append(term_value)
        self.rebuild(self.terms)

    def delete_term(self, logical_name: str) -> None:
        self.terms = [term for term in self.terms if term.logical_name != logical_name]
        self.rebuild(self.terms)


def create_embedding_provider(config: AppConfig, repository: StandardRepository) -> EmbeddingProvider:
    """환경 설정에 맞는 embedding provider를 생성합니다."""

    if config.openai_api_key and OpenAIEmbeddings is not None:
        with contextlib.suppress(Exception):
            return LangChainOpenAIEmbeddingProvider(config)
    return HashingEmbeddingProvider(repository.list_words())


def create_vector_store(config: AppConfig, embedding_provider: EmbeddingProvider, terms: Sequence[StandardTerm]) -> TermVectorStore:
    """FAISS 사용 가능 여부에 따라 Vector DB 구현체를 생성합니다."""

    try:
        store = FaissTermVectorStore(embedding_provider, terms)
        LOGGER.info("FAISS vector store initialized with %s terms.", len(terms))
        return store
    except Exception as exc:
        LOGGER.info("FAISS 초기화 실패로 in-memory vector store를 사용합니다: %s", exc)
        return InMemoryTermVectorStore(embedding_provider, terms)


def term_to_vector_text(term_value: StandardTerm) -> str:
    """표준용어를 embedding 입력 텍스트로 변환합니다."""

    return f"표준용어명: {term_value.logical_name}\n도메인: {term_value.domain}\n설명: {term_value.description}"


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """두 벡터의 cosine similarity를 계산합니다."""

    if not left or not right:
        return 0.0
    dot = sum(l_value * r_value for l_value, r_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


# -----------------------------------------------------------------------------
# 텍스트 정규화와 표준단어 매칭 유틸리티입니다.
# 한국어 복합명은 공백 없이 붙어 있을 수 있으므로 compact 문자열과 token 양쪽을 모두 사용합니다.
# -----------------------------------------------------------------------------
def normalize_text(text_value: str) -> str:
    """공백과 일부 문장부호를 제거한 비교용 문자열을 반환합니다."""

    return re.sub(r"[\s'\"`“”‘’.,!?]", "", text_value or "").lower()


def tokenize(text_value: str) -> list[str]:
    """한국어/영문/숫자 토큰을 추출합니다."""

    return [token.lower() for token in re.findall(r"[가-힣A-Za-z0-9]+", text_value or "")]


def word_aliases(word: StandardWord) -> list[str]:
    """표준단어 논리명, 물리명, 동의어를 하나의 alias 목록으로 반환합니다."""

    aliases = [word.logical_name, word.physical_name, *word.synonyms]
    return [alias for alias in aliases if alias]


def expand_tokens_with_standard_words(text_value: str, words: Sequence[StandardWord]) -> list[str]:
    """텍스트 토큰에 표준단어 alias 매칭 결과를 추가해 embedding 품질을 높입니다."""

    compact = normalize_text(text_value)
    tokens = tokenize(text_value)
    expanded = list(tokens)
    for word in words:
        for alias in word_aliases(word):
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in compact:
                expanded.extend([word.logical_name.lower()] * 3)
                break
    return expanded


def is_too_short_or_ambiguous(description: str, words: Sequence[StandardWord]) -> tuple[bool, list[str]]:
    """등록/추천 대상 설명이 너무 짧거나 모호한지 판단합니다."""

    reasons: list[str] = []
    compact = normalize_text(description)
    if len(compact) < 6:
        reasons.append("설명이 너무 짧습니다.")
    matched_words = [
        word.logical_name
        for word in words
        if any(normalize_text(alias) and normalize_text(alias) in compact for alias in word_aliases(word))
    ]
    if len(set(matched_words)) < 2:
        reasons.append("표준단어로 해석할 단서가 부족합니다.")
    has_class_word = any(word.word_type == "class" and word.logical_name in matched_words for word in words)
    if not has_class_word and not any(token in compact for token in ["시점", "시간", "날짜", "일시", "수량", "번호", "코드", "식별"]):
        reasons.append("속성 분류어를 추정하기 어렵습니다.")
    return bool(reasons), reasons


def decompose_logical_name(logical_name: str, words: Sequence[StandardWord]) -> list[StandardWord] | None:
    """표준용어 논리명이 표준단어 조합으로만 구성되는지 동적 계획법으로 분해합니다."""

    ordered_words = sorted(words, key=lambda item: len(item.logical_name), reverse=True)
    memo: dict[int, list[StandardWord] | None] = {}

    def walk(position: int) -> list[StandardWord] | None:
        if position == len(logical_name):
            return []
        if position in memo:
            return memo[position]
        for word in ordered_words:
            if logical_name.startswith(word.logical_name, position):
                tail = walk(position + len(word.logical_name))
                if tail is not None:
                    memo[position] = [word, *tail]
                    return memo[position]
        memo[position] = None
        return None

    return walk(0)


# -----------------------------------------------------------------------------
# 외부 지식 검색 도구입니다.
# Tavily가 설정되어 있으면 request_agent에서 RDB/Vector DB 결과와 함께 웹 검색 요약 근거로 사용합니다.
# MCP를 붙이는 경우에도 ExternalKnowledgeProvider 프로토콜을 구현하면 같은 위치에 연결할 수 있습니다.
# -----------------------------------------------------------------------------
class ExternalKnowledgeProvider(Protocol):
    """외부 검색 또는 MCP 검색 도구 인터페이스입니다."""

    def search(self, query: str) -> list[dict[str, Any]]:
        """질의와 관련된 외부 지식 검색 결과를 반환합니다."""


class NoOpKnowledgeProvider:
    """외부 검색 설정이 없을 때 사용하는 no-op provider입니다."""

    def search(self, query: str) -> list[dict[str, Any]]:
        return []


class TavilyKnowledgeProvider:
    """Tavily Search API 기반 외부 검색 provider입니다."""

    def __init__(self, api_key: str) -> None:
        if TavilySearchResults is None:
            raise RuntimeError("langchain-community TavilySearchResults를 사용할 수 없습니다.")
        os.environ.setdefault("TAVILY_API_KEY", api_key)
        self._tool = TavilySearchResults(max_results=3)

    def search(self, query: str) -> list[dict[str, Any]]:
        try:
            raw_results = self._tool.invoke({"query": query})
            if isinstance(raw_results, list):
                return [cast(dict[str, Any], item) for item in raw_results]
        except Exception as exc:
            LOGGER.warning("Tavily 검색 실패: %s", exc)
        return []


def create_external_knowledge_provider(config: AppConfig) -> ExternalKnowledgeProvider:
    """Tavily API 키가 있으면 외부 검색 provider를 생성합니다."""

    if config.tavily_api_key and TavilySearchResults is not None:
        with contextlib.suppress(Exception):
            return TavilyKnowledgeProvider(config.tavily_api_key)
    return NoOpKnowledgeProvider()


# -----------------------------------------------------------------------------
# RuntimeServices는 모든 Agent가 공유하는 의존성을 담습니다.
# 이 객체만 교체하면 Mock에서 Oracle/FAISS/OpenAI 운영 환경으로 전환할 수 있습니다.
# -----------------------------------------------------------------------------
@dataclass
class RuntimeServices:
    """Agent 실행에 필요한 공유 서비스 모음입니다."""

    config: AppConfig
    repository: StandardRepository
    vector_store: TermVectorStore
    embedding_provider: EmbeddingProvider
    knowledge_provider: ExternalKnowledgeProvider
    tracer: Any
    llm: Any | None


def create_llm(config: AppConfig) -> Any | None:
    """OpenAI Chat 모델을 생성합니다. 설정이 없으면 None을 반환합니다."""

    if ChatOpenAI is None or not config.openai_api_key:
        return None
    kwargs: dict[str, Any] = {
        "model": config.llm_model,
        "api_key": config.openai_api_key,
        "temperature": 0,
    }
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    with contextlib.suppress(Exception):
        return ChatOpenAI(**kwargs)
    return None


def create_repository(config: AppConfig) -> StandardRepository:
    """설정에 따라 Mock 또는 Oracle 저장소를 생성합니다."""

    if config.use_mock_data:
        return MockStandardRepository()
    return OracleStandardRepository(config)


def create_runtime_services() -> RuntimeServices:
    """환경 설정을 읽어 Agent runtime 서비스를 생성합니다."""

    config = AppConfig.from_env()
    configure_langsmith(config)
    tracer = configure_opentelemetry(config)
    repository = create_repository(config)
    embedding_provider = create_embedding_provider(config, repository)
    vector_store = create_vector_store(config, embedding_provider, repository.list_terms())
    knowledge_provider = create_external_knowledge_provider(config)
    return RuntimeServices(
        config=config,
        repository=repository,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        knowledge_provider=knowledge_provider,
        tracer=tracer,
        llm=create_llm(config),
    )


# -----------------------------------------------------------------------------
# router_agent입니다.
# LLM 사용 가능 시 JSON schema 기반 분류를 시도하고, 실패하거나 Mock 환경이면 heuristic 분류를 사용합니다.
# 어떤 경우에도 IntentDecision JSON 문자열을 state.classification_json에 남깁니다.
# -----------------------------------------------------------------------------
class RouterAgent:
    """사용자 입력을 요구사항의 intent 중 하나로 분류하는 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    def run(self, state: AgentState) -> dict[str, Any]:
        """LangGraph router_agent node handler입니다."""

        user_input = state.get("user_input") or last_user_message(state)
        with self.services.tracer.start_as_current_span("router_agent") as span:
            decision = self.classify(user_input, state)
            decision_json = to_json(model_to_dict(decision))
            span.set_attribute("intent", decision.intent)
            span.set_attribute("confidence", decision.confidence)
            span.set_attribute("reason", decision.reason)
            return {
                "intent": decision.intent,
                "classification_json": decision_json,
                "route_reason": decision.reason,
                "logs": [
                    make_log(
                        "router_agent",
                        "intent_classified",
                        {
                            "classification_json": decision_json,
                            "user_input": user_input,
                        },
                    )
                ],
            }

    def classify(self, user_input: str, state: AgentState) -> IntentDecision:
        """LLM 또는 heuristic으로 intent를 분류합니다."""

        forced_intent = state.get("forced_intent")
        if forced_intent in SUPPORTED_INTENTS:
            return IntentDecision(
                intent=forced_intent,
                confidence=1.0,
                reason="UI 액션 또는 상위 호출에서 의도가 명시되었습니다.",
                normalized_input=normalize_text(user_input),
            )

        llm_decision = self._classify_with_llm(user_input, state)
        if llm_decision is not None:
            return llm_decision
        return self._classify_with_rules(user_input, state)

    def _classify_with_llm(self, user_input: str, state: AgentState) -> IntentDecision | None:
        """OpenAI gpt-5가 설정된 경우 JSON schema 기반 intent 분류를 수행합니다."""

        if self.services.llm is None:
            return None
        system_prompt = (
            "너는 표준용어 관리 Multi Agent의 router_agent다. "
            "사용자 입력을 meaning_search, term_recommend, term_register, term_change, request 중 하나로 분류한다. "
            "출력은 반드시 JSON이며 내부 추론 과정은 쓰지 말고 짧은 reason만 제공한다. "
            "용어 설명만 입력되면 meaning_search, 추천 요청은 term_recommend, 신청/등록은 term_register, "
            "수정/삭제/변경은 term_change, 일반 문의는 request다."
        )
        context = {
            "search_count": state.get("search_count", 0),
            "has_recommendations": bool(state.get("recommendations")),
            "last_description": state.get("last_description", ""),
        }
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"입력: {user_input}\n대화상태: {to_json(context)}"),
        ]
        try:
            if hasattr(self.services.llm, "with_structured_output"):
                structured = self.services.llm.with_structured_output(IntentDecision)
                result = structured.invoke(messages)
                if isinstance(result, IntentDecision):
                    return result
                if isinstance(result, dict):
                    return IntentDecision(**result)
            raw = self.services.llm.invoke(messages)
            content = getattr(raw, "content", str(raw))
            parsed = json.loads(str(content))
            return IntentDecision(**parsed)
        except Exception as exc:
            LOGGER.info("LLM intent 분류 실패, rule fallback 사용: %s", exc)
            return None

    def _classify_with_rules(self, user_input: str, state: AgentState) -> IntentDecision:
        """Mock 실행을 위한 규칙 기반 intent 분류입니다."""

        compact = normalize_text(user_input)
        query_words = ["이란", "무엇", "뭔", "있나요", "알려", "문의", "체크리스트", "?", "뜻"]
        change_words = ["변경", "수정", "삭제", "바꿔", "정정", "지워"]
        recommend_words = ["추천", "제안", "만들어"]
        register_words = ["신청", "등록", "반영"]

        if any(word in compact for word in change_words):
            intent: IntentName = "term_change"
            reason = "변경/수정/삭제 지시가 포함되었습니다."
            confidence = 0.94
        elif any(word in compact for word in register_words) and (
            state.get("recommendations") or state.get("selected_term")
        ):
            intent = "term_register"
            reason = "추천 용어 신청 또는 등록 요청으로 판단했습니다."
            confidence = 0.9
        elif any(word in compact for word in recommend_words):
            intent = "term_recommend"
            reason = "사용자가 신규 용어 추천을 요청했습니다."
            confidence = 0.92
        elif any(word in user_input for word in query_words):
            intent = "request"
            reason = "질문형 표현이 포함되어 문의로 판단했습니다."
            confidence = 0.87
        elif self._looks_like_short_dictionary_query(user_input, compact):
            intent = "request"
            reason = "짧은 단어 단독 입력으로 용어 문의로 판단했습니다."
            confidence = 0.78
        else:
            intent = "meaning_search"
            reason = "용어 설명을 기반으로 한 의미 검색 요청으로 판단했습니다."
            confidence = 0.82

        return IntentDecision(
            intent=intent,
            confidence=confidence,
            reason=reason,
            normalized_input=compact,
        )

    def _looks_like_short_dictionary_query(self, user_input: str, compact: str) -> bool:
        """'재공'처럼 짧은 단어 단독 입력을 문의로 해석합니다."""

        if len(tokenize(user_input)) > 1:
            return False
        if len(compact) > 6:
            return False
        if not compact:
            return False
        words = self.services.repository.find_words(compact)
        terms = self.services.repository.search_terms(compact, limit=1)
        exact_word = any(normalize_text(word.logical_name) == compact for word in words)
        exact_term = any(normalize_text(term.logical_name) == compact for term in terms)
        return exact_word or exact_term


# -----------------------------------------------------------------------------
# select_agent입니다.
# 사용자 설명을 embedding으로 변환한 뒤 Vector DB에서 유사도 상위 3건을 조회합니다.
# similarity score가 0.6 이상인 결과만 사용자 응답에 표시합니다.
# -----------------------------------------------------------------------------
class SelectAgent:
    """표준용어 의미 검색 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    def run(self, state: AgentState) -> dict[str, Any]:
        """LangGraph select_agent node handler입니다."""

        description = state.get("active_description") or state.get("user_input") or last_user_message(state)
        with self.services.tracer.start_as_current_span("select_agent") as span:
            too_short, reasons = is_too_short_or_ambiguous(description, self.services.repository.list_words())
            if too_short and len(normalize_text(description)) < 4:
                response = "설명이 너무 짧거나 모호합니다. 대상, 행위, 속성이 드러나도록 조금 더 구체적으로 입력해 주세요."
                span.set_attribute("search.skipped", True)
                return {
                    "response": response,
                    "logs": [make_log("select_agent", "clarification_required", {"reasons": reasons})],
                    "messages": [AIMessage(content=response)],
                }

            results = self.services.vector_store.search(description, top_k=SIMILARITY_TOP_K)
            all_results = [model_to_dict(item) for item in results]
            passed = [item for item in results if item.similarity >= SIMILARITY_THRESHOLD]
            response = self._format_response(passed)
            span.set_attribute("search.query", description)
            span.set_attribute("search.top_k", SIMILARITY_TOP_K)
            span.set_attribute("search.threshold", SIMILARITY_THRESHOLD)
            span.set_attribute("search.passed_count", len(passed))
            return {
                "response": response,
                "term_results": all_results,
                "last_search_results": all_results,
                "last_description": description,
                "search_count": int(state.get("search_count", 0)) + 1,
                "logs": [
                    make_log(
                        "select_agent",
                        "vector_search_completed",
                        {
                            "query": description,
                            "top_k": SIMILARITY_TOP_K,
                            "threshold": SIMILARITY_THRESHOLD,
                            "results": all_results,
                        },
                    )
                ],
                "messages": [AIMessage(content=response)],
            }

    def _format_response(self, results: Sequence[SimilarTermResult]) -> str:
        """검색 결과를 사용자 출력 형식으로 변환합니다."""

        if not results:
            return "유사도 기준을 만족하는 표준용어를 찾지 못했습니다. 조회 후 필요하면 신규 용어 추천을 요청할 수 있습니다."
        lines = [
            f"{item.logical_name}: {item.description} (similarity={item.similarity:.4f})"
            for item in results
        ]
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# recommend_agent입니다.
# Tree of Thoughts 방식으로 여러 표준단어 조합 후보를 만들고, Self-Correction 단계에서
# 중복/분류어 누락/표준단어 미분해 후보를 제거합니다.
# -----------------------------------------------------------------------------
class TreeOfThoughtsTermComposer:
    """표준단어 기반 신규 표준용어 후보 생성기입니다."""

    def generate(
        self,
        description: str,
        words: Sequence[StandardWord],
        existing_terms: Sequence[StandardTerm],
        max_candidates: int = 3,
    ) -> list[TermProposal]:
        """설명과 표준단어를 조합해 신규 표준용어 후보를 생성합니다."""

        matched_words = self._match_words_by_position(description, words)
        inferred_words = self._infer_missing_class_words(description, matched_words, words)
        base_path = self._deduplicate_words([*matched_words, *inferred_words])
        thought_paths = self._build_thought_paths(base_path, description, words)
        existing_names = {term.logical_name for term in existing_terms}

        proposals: list[TermProposal] = []
        for path in thought_paths:
            corrected = self._self_correct_path(path, words)
            if not corrected:
                continue
            logical_name = "".join(word.logical_name for word in corrected)
            if logical_name in existing_names:
                continue
            if decompose_logical_name(logical_name, words) is None:
                continue
            score = self._score_path(corrected, description)
            proposals.append(
                TermProposal(
                    logical_name=logical_name,
                    used_words=[word.logical_name for word in corrected],
                    score=round(score, 4),
                    reason="설명 내 표준단어 alias와 속성 분류어를 조합했습니다.",
                )
            )

        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals[:max_candidates]

    def _match_words_by_position(self, description: str, words: Sequence[StandardWord]) -> list[StandardWord]:
        """설명에서 alias가 발견된 순서대로 표준단어를 정렬합니다."""

        compact = normalize_text(description)
        positioned: list[tuple[int, int, StandardWord]] = []
        for index, word in enumerate(words):
            positions: list[int] = []
            for alias in word_aliases(word):
                alias_norm = normalize_text(alias)
                if alias_norm and alias_norm in compact:
                    positions.append(compact.find(alias_norm))
            if positions:
                positioned.append((min(positions), index, word))
        positioned.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in positioned]

    def _infer_missing_class_words(
        self,
        description: str,
        matched_words: Sequence[StandardWord],
        words: Sequence[StandardWord],
    ) -> list[StandardWord]:
        """설명 단서로 누락된 속성 분류어를 보강합니다."""

        if matched_words and matched_words[-1].word_type == "class":
            return []
        compact = normalize_text(description)
        rules = [
            (["일시", "시점", "시간", "timestamp", "datetime"], "일시"),
            (["날짜", "일자", "date"], "일자"),
            (["수량", "개수", "양", "quantity"], "수량"),
            (["번호", "number", "no"], "번호"),
            (["코드", "code"], "코드"),
            (["식별", "identifier", "id", "아이디"], "ID"),
        ]
        word_by_name = {word.logical_name: word for word in words}
        for tokens, logical_name in rules:
            if any(token.lower() in compact for token in tokens) and logical_name in word_by_name:
                return [word_by_name[logical_name]]
        return []

    def _build_thought_paths(
        self,
        base_path: Sequence[StandardWord],
        description: str,
        words: Sequence[StandardWord],
    ) -> list[list[StandardWord]]:
        """Tree of Thoughts 후보 경로를 생성합니다."""

        if not base_path:
            return []
        paths = [list(base_path)]
        word_by_name = {word.logical_name: word for word in words}
        names = [word.logical_name for word in base_path]
        compact = normalize_text(description)

        if "종료" in compact or "완료" in compact or "마감" in compact:
            if "시작" in names and "종료" in word_by_name:
                paths.append([word_by_name["종료"] if word.logical_name == "시작" else word for word in base_path])
            elif "종료" not in names and "종료" in word_by_name:
                paths.append(self._insert_before_class(base_path, word_by_name["종료"]))

        if "시작" in compact or "개시" in compact or "착수" in compact or "start" in compact:
            if "종료" in names and "시작" in word_by_name:
                paths.append([word_by_name["시작"] if word.logical_name == "종료" else word for word in base_path])

        return [self._deduplicate_words(path) for path in paths]

    def _insert_before_class(self, path: Sequence[StandardWord], word: StandardWord) -> list[StandardWord]:
        """속성 분류어 앞에 업무 단어를 삽입합니다."""

        output = list(path)
        for index, item in enumerate(output):
            if item.word_type == "class":
                output.insert(index, word)
                return output
        output.append(word)
        return output

    def _self_correct_path(self, path: Sequence[StandardWord], words: Sequence[StandardWord]) -> list[StandardWord]:
        """후보 경로의 중복과 분류어 위치를 교정합니다."""

        corrected = self._deduplicate_words(path)
        if not corrected:
            return []
        class_words = [word for word in corrected if word.word_type == "class"]
        business_words = [word for word in corrected if word.word_type != "class"]
        if not class_words:
            fallback = next((word for word in words if word.logical_name == "일시"), None)
            if fallback is not None:
                class_words = [fallback]
        if not business_words or not class_words:
            return []
        return [*business_words, class_words[-1]]

    def _deduplicate_words(self, path: Sequence[StandardWord]) -> list[StandardWord]:
        """표준단어 순서를 유지하면서 중복을 제거합니다."""

        seen: set[str] = set()
        output: list[StandardWord] = []
        for word in path:
            if word.logical_name not in seen:
                seen.add(word.logical_name)
                output.append(word)
        return output

    def _score_path(self, path: Sequence[StandardWord], description: str) -> float:
        """후보 경로를 설명 coverage와 명명 규칙 기준으로 점수화합니다."""

        compact = normalize_text(description)
        coverage = 0.0
        for word in path:
            if any(normalize_text(alias) and normalize_text(alias) in compact for alias in word_aliases(word)):
                coverage += 1.0
        class_bonus = 1.0 if path and path[-1].word_type == "class" else 0.0
        length_penalty = max(0, len(path) - 4) * 0.1
        return coverage + class_bonus - length_penalty


class RecommendAgent:
    """사용자 설명과 RDB 표준단어를 조합해 신규 표준용어를 추천하는 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.composer = TreeOfThoughtsTermComposer()

    def run(self, state: AgentState) -> dict[str, Any]:
        """LangGraph recommend_agent node handler입니다."""

        with self.services.tracer.start_as_current_span("recommend_agent") as span:
            if int(state.get("search_count", 0)) < 1:
                response = "추천 전에 용어 설명 기반 조회를 1회 이상 수행해 주세요."
                span.set_attribute("recommend.skipped", True)
                return {
                    "response": response,
                    "logs": [make_log("recommend_agent", "search_required_before_recommendation")],
                    "messages": [AIMessage(content=response)],
                }

            description = (
                state.get("active_description")
                or state.get("last_description")
                or state.get("user_input")
                or last_user_message(state)
            )
            too_short, reasons = is_too_short_or_ambiguous(description, self.services.repository.list_words())
            if too_short:
                response = "추천할 수 없습니다. " + " ".join(reasons)
                return {
                    "response": response,
                    "logs": [make_log("recommend_agent", "clarification_required", {"reasons": reasons})],
                    "messages": [AIMessage(content=response)],
                }

            proposals = self.composer.generate(
                description=description,
                words=self.services.repository.list_words(),
                existing_terms=self.services.repository.list_terms(),
            )
            proposal_dicts = [model_to_dict(item) for item in proposals]
            response = "\n".join(item.logical_name for item in proposals) if proposals else "추천 가능한 신규 표준용어를 찾지 못했습니다."
            span.set_attribute("recommend.proposal_count", len(proposals))
            return {
                "response": response,
                "recommendations": proposal_dicts,
                "active_description": description,
                "logs": [
                    make_log(
                        "recommend_agent",
                        "tot_self_correction_completed",
                        {"description": description, "proposals": proposal_dicts},
                    )
                ],
                "messages": [AIMessage(content=response)],
            }


# -----------------------------------------------------------------------------
# validate_agent입니다.
# 신규 등록과 변경 요청을 분리 검증합니다. 설명이 짧거나 모호하면 등록/변경 실행을 막습니다.
# -----------------------------------------------------------------------------
class TermValidator:
    """표준용어 등록/변경 검증 규칙 모음입니다."""

    def __init__(self, repository: StandardRepository, vector_store: TermVectorStore) -> None:
        self.repository = repository
        self.vector_store = vector_store

    def validate_new_term(self, logical_name: str, description: str) -> ValidationReport:
        """신규 표준용어 신청 가능 여부를 검증합니다."""

        reasons: list[str] = []
        words = self.repository.list_words()
        if self.repository.get_term(logical_name):
            reasons.append("이미 등록된 표준용어입니다.")
        if not re.fullmatch(r"[가-힣A-Za-z0-9_]+", logical_name or ""):
            reasons.append("논리명에는 한글, 영문, 숫자, 밑줄만 사용할 수 있습니다.")
        decomposed = decompose_logical_name(logical_name, words)
        if not decomposed:
            reasons.append("표준단어 논리명 조합으로 분해되지 않습니다.")
        elif decomposed[-1].word_type != "class":
            reasons.append("마지막 표준단어는 속성 분류어여야 합니다.")

        too_short, description_reasons = is_too_short_or_ambiguous(description, words)
        if too_short:
            reasons.extend(description_reasons)

        duplicate_meaning = [
            result
            for result in self.vector_store.search(description, top_k=1)
            if result.similarity >= 0.95
        ]
        if duplicate_meaning:
            reasons.append(f"기존 표준용어와 의미가 거의 같습니다: {duplicate_meaning[0].logical_name}")

        return ValidationReport(
            ok=not reasons,
            reasons=reasons,
            target_intent="term_register",
            logical_name=logical_name,
            description=description,
        )

    def validate_change(self, plan: ChangePlan) -> ValidationReport:
        """표준용어 수정/삭제 가능 여부를 검증합니다."""

        reasons: list[str] = []
        current = self.repository.get_term(plan.logical_name)
        if current is None:
            reasons.append("변경 대상 표준용어가 RDB에 없습니다.")

        if plan.action == "update":
            if not plan.new_description:
                reasons.append("수정할 설명이 비어 있습니다.")
            elif current and current.description.strip() == plan.new_description.strip():
                reasons.append("기존 설명과 동일합니다.")
            else:
                too_short, description_reasons = is_too_short_or_ambiguous(
                    plan.new_description or "",
                    self.repository.list_words(),
                )
                if too_short:
                    reasons.extend(description_reasons)

        return ValidationReport(
            ok=not reasons,
            reasons=reasons,
            target_intent="term_change",
            logical_name=plan.logical_name,
            description=plan.new_description,
        )


class ValidateAgent:
    """등록/변경 전 업무 규칙을 검증하는 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.validator = TermValidator(services.repository, services.vector_store)

    def run(self, state: AgentState) -> dict[str, Any]:
        """LangGraph validate_agent node handler입니다."""

        intent = state.get("intent")
        with self.services.tracer.start_as_current_span("validate_agent") as span:
            if intent == "term_register":
                return self._validate_register(state, span)
            if intent == "term_change":
                return self._validate_change(state, span)
            response = "검증 대상 의도가 아닙니다."
            return {
                "response": response,
                "validation": model_to_dict(
                    ValidationReport(ok=False, reasons=[response], target_intent="request")
                ),
                "logs": [make_log("validate_agent", "unsupported_intent", {"intent": intent})],
                "messages": [AIMessage(content=response)],
            }

    def _validate_register(self, state: AgentState, span: Any) -> dict[str, Any]:
        """신규 등록 요청을 검증합니다."""

        selected_term = state.get("selected_term") or self._first_recommendation_name(state)
        description = state.get("active_description") or state.get("last_description") or state.get("user_input", "")
        report = self.validator.validate_new_term(selected_term, description)
        report_dict = model_to_dict(report)
        span.set_attribute("validation.ok", report.ok)
        span.set_attribute("validation.logical_name", selected_term)
        if not report.ok:
            response = "등록이 불가합니다. " + " ".join(report.reasons)
            return {
                "response": response,
                "validation": report_dict,
                "logs": [make_log("validate_agent", "register_validation_failed", report_dict)],
                "messages": [AIMessage(content=response)],
            }
        return {
            "validation": report_dict,
            "selected_term": selected_term,
            "active_description": description,
            "logs": [make_log("validate_agent", "register_validation_passed", report_dict)],
        }

    def _validate_change(self, state: AgentState, span: Any) -> dict[str, Any]:
        """변경 요청을 파싱하고 검증합니다."""

        plan = self._load_or_parse_change_plan(state)
        if plan is None:
            response = "변경 대상 표준용어와 변경 내용을 확인할 수 없습니다."
            report = ValidationReport(ok=False, reasons=[response], target_intent="term_change")
            return {
                "response": response,
                "validation": model_to_dict(report),
                "logs": [make_log("validate_agent", "change_parse_failed", {"input": state.get("user_input")})],
                "messages": [AIMessage(content=response)],
            }

        report = self.validator.validate_change(plan)
        report_dict = model_to_dict(report)
        span.set_attribute("validation.ok", report.ok)
        span.set_attribute("change.action", plan.action)
        span.set_attribute("change.logical_name", plan.logical_name)
        if not report.ok:
            response = "변경이 불가합니다. " + " ".join(report.reasons)
            return {
                "response": response,
                "validation": report_dict,
                "pending_change_request": model_to_dict(plan),
                "logs": [make_log("validate_agent", "change_validation_failed", report_dict)],
                "messages": [AIMessage(content=response)],
            }
        if not state.get("apply_change"):
            response = "변경이 가능합니다."
            return {
                "response": response,
                "validation": report_dict,
                "pending_change_request": model_to_dict(plan),
                "logs": [make_log("validate_agent", "change_validation_passed_waiting_confirm", model_to_dict(plan))],
                "messages": [AIMessage(content=response)],
            }
        return {
            "validation": report_dict,
            "pending_change_request": model_to_dict(plan),
            "logs": [make_log("validate_agent", "change_validation_passed", model_to_dict(plan))],
        }

    def _first_recommendation_name(self, state: AgentState) -> str:
        """state의 첫 추천 후보명을 반환합니다."""

        recommendations = state.get("recommendations") or []
        if recommendations:
            return str(recommendations[0].get("logical_name", ""))
        return normalize_text(state.get("user_input", ""))

    def _load_or_parse_change_plan(self, state: AgentState) -> ChangePlan | None:
        """pending change가 있으면 사용하고, 없으면 사용자 입력에서 변경 계획을 파싱합니다."""

        pending = state.get("pending_change_request")
        if pending and state.get("apply_change"):
            with contextlib.suppress(Exception):
                return ChangePlan(**pending)
        return parse_change_request(state.get("user_input") or last_user_message(state), self.services.repository)


def parse_change_request(user_input: str, repository: StandardRepository) -> ChangePlan | None:
    """사용자의 자연어 변경 요청을 update/delete ChangePlan으로 변환합니다."""

    quoted = re.findall(r"['\"“”‘’]([^'\"“”‘’]+)['\"“”‘’]", user_input)
    compact = normalize_text(user_input)
    known_terms = sorted(repository.list_terms(), key=lambda term: len(term.logical_name), reverse=True)
    logical_name = quoted[0] if quoted else ""
    if not logical_name:
        logical_name = next((term.logical_name for term in known_terms if normalize_text(term.logical_name) in compact), "")
    if not logical_name:
        return None

    if any(token in compact for token in ["삭제", "지워", "제거"]):
        return ChangePlan(action="delete", logical_name=logical_name, requested_text=user_input)

    new_description = quoted[1] if len(quoted) >= 2 else ""
    if not new_description:
        patterns = [
            r"설명을\s*(.+?)(?:로|으로)\s*(?:변경|수정|바꿔)",
            r"설명\s*[:=]\s*(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_input)
            if match:
                new_description = match.group(1).strip()
                break
    if not new_description:
        return None
    return ChangePlan(
        action="update",
        logical_name=logical_name,
        new_description=new_description,
        requested_text=user_input,
    )


# -----------------------------------------------------------------------------
# change_agent입니다.
# validate_agent 통과 후 RDB와 Vector DB를 함께 변경합니다.
# Vector DB 반영 실패 시 가능한 범위에서 RDB 보상 처리를 시도합니다.
# -----------------------------------------------------------------------------
class ChangeAgent:
    """표준용어 insert/update/delete를 수행하는 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    def run(self, state: AgentState) -> dict[str, Any]:
        """LangGraph change_agent node handler입니다."""

        intent = state.get("intent")
        with self.services.tracer.start_as_current_span("change_agent") as span:
            if intent == "term_register":
                return self._register_term(state, span)
            if intent == "term_change":
                return self._apply_change(state, span)
            response = "변경 처리 대상 의도가 아닙니다."
            return {
                "response": response,
                "logs": [make_log("change_agent", "unsupported_intent", {"intent": intent})],
                "messages": [AIMessage(content=response)],
            }

    def _register_term(self, state: AgentState, span: Any) -> dict[str, Any]:
        """검증된 신규 표준용어를 RDB와 Vector DB에 등록합니다."""

        logical_name = state.get("selected_term", "")
        description = state.get("active_description") or state.get("last_description") or ""
        term_value = StandardTerm(logical_name=logical_name, description=description, domain=DEFAULT_DOMAIN)
        inserted = False
        try:
            self.services.repository.insert_term(term_value)
            inserted = True
            self.services.vector_store.upsert_term(term_value)
            response = "등록이 완료되었습니다."
            span.set_attribute("register.logical_name", logical_name)
            return {
                "response": response,
                "change_result": {"ok": True, "action": "insert", "logical_name": logical_name},
                "logs": [make_log("change_agent", "register_completed", model_to_dict(term_value))],
                "messages": [AIMessage(content=response)],
            }
        except Exception as exc:
            if inserted:
                with contextlib.suppress(Exception):
                    self.services.repository.delete_term(logical_name)
                    self.services.vector_store.delete_term(logical_name)
            span.record_exception(exc)
            LOGGER.exception("표준용어 등록 실패")
            response = "등록 시 오류가 발생해 취소 처리되었습니다."
            return {
                "response": response,
                "change_result": {"ok": False, "action": "insert", "logical_name": logical_name, "error": str(exc)},
                "logs": [make_log("change_agent", "register_failed", {"logical_name": logical_name, "error": str(exc)})],
                "messages": [AIMessage(content=response)],
            }

    def _apply_change(self, state: AgentState, span: Any) -> dict[str, Any]:
        """검증된 표준용어 변경 요청을 RDB와 Vector DB에 반영합니다."""

        plan = ChangePlan(**state.get("pending_change_request", {}))
        old_term = self.services.repository.get_term(plan.logical_name)
        try:
            if plan.action == "delete":
                self.services.repository.delete_term(plan.logical_name)
                self.services.vector_store.delete_term(plan.logical_name)
            else:
                assert plan.new_description is not None
                self.services.repository.update_term_description(plan.logical_name, plan.new_description)
                updated = StandardTerm(
                    logical_name=plan.logical_name,
                    description=plan.new_description,
                    domain=old_term.domain if old_term else DEFAULT_DOMAIN,
                )
                self.services.vector_store.upsert_term(updated)
            response = "변경이 완료되었습니다."
            span.set_attribute("change.action", plan.action)
            span.set_attribute("change.logical_name", plan.logical_name)
            return {
                "response": response,
                "change_result": {"ok": True, **model_to_dict(plan)},
                "pending_change_request": {},
                "logs": [make_log("change_agent", "change_completed", model_to_dict(plan))],
                "messages": [AIMessage(content=response)],
            }
        except Exception as exc:
            if old_term is not None:
                with contextlib.suppress(Exception):
                    self.services.repository.update_term_description(old_term.logical_name, old_term.description)
                    self.services.vector_store.upsert_term(old_term)
            span.record_exception(exc)
            LOGGER.exception("표준용어 변경 실패")
            response = "변경 시 오류가 발생해 취소 처리되었습니다."
            return {
                "response": response,
                "change_result": {"ok": False, **model_to_dict(plan), "error": str(exc)},
                "logs": [make_log("change_agent", "change_failed", {"plan": model_to_dict(plan), "error": str(exc)})],
                "messages": [AIMessage(content=response)],
            }


# -----------------------------------------------------------------------------
# request_agent입니다.
# 일반 문의는 RDB, Vector DB, 선택적 Tavily 검색 결과를 모아 요약합니다.
# LLM이 없으면 deterministic 요약으로 답변합니다.
# -----------------------------------------------------------------------------
class RequestAgent:
    """일반 문의 응답 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services

    def run(self, state: AgentState) -> dict[str, Any]:
        """LangGraph request_agent node handler입니다."""

        query = state.get("user_input") or last_user_message(state)
        with self.services.tracer.start_as_current_span("request_agent") as span:
            word_matches = self.services.repository.find_words(query)
            term_matches = self.services.repository.search_terms(query, limit=5)
            vector_matches = self.services.vector_store.search(query, top_k=SIMILARITY_TOP_K)
            web_matches = self.services.knowledge_provider.search(query)
            response = self._summarize(query, word_matches, term_matches, vector_matches, web_matches)
            span.set_attribute("request.word_count", len(word_matches))
            span.set_attribute("request.term_count", len(term_matches))
            span.set_attribute("request.vector_count", len(vector_matches))
            span.set_attribute("request.web_count", len(web_matches))
            return {
                "response": response,
                "logs": [
                    make_log(
                        "request_agent",
                        "request_answered",
                        {
                            "query": query,
                            "word_matches": [model_to_dict(item) for item in word_matches],
                            "term_matches": [model_to_dict(item) for item in term_matches],
                            "vector_matches": [model_to_dict(item) for item in vector_matches],
                            "web_matches": web_matches,
                        },
                    )
                ],
                "messages": [AIMessage(content=response)],
            }

    def _summarize(
        self,
        query: str,
        word_matches: Sequence[StandardWord],
        term_matches: Sequence[StandardTerm],
        vector_matches: Sequence[SimilarTermResult],
        web_matches: Sequence[dict[str, Any]],
    ) -> str:
        """검색 근거를 요약합니다."""

        llm_response = self._summarize_with_llm(query, word_matches, term_matches, vector_matches, web_matches)
        if llm_response:
            return llm_response

        lines: list[str] = []
        for word in word_matches:
            description = word.description or ", ".join(word.synonyms)
            lines.append(f"{word.logical_name}: {description}")
        for term in term_matches:
            lines.append(f"{term.logical_name}: {term.description}")
        for result in vector_matches:
            if result.similarity >= SIMILARITY_THRESHOLD and result.logical_name not in "\n".join(lines):
                lines.append(f"{result.logical_name}: {result.description}")
        for item in web_matches[:2]:
            title = item.get("title") or item.get("url") or "외부 검색 결과"
            content = item.get("content") or item.get("snippet") or ""
            if content:
                lines.append(f"{title}: {content}")
        if not lines:
            return "관련 표준단어, 표준용어, 외부 검색 결과를 찾지 못했습니다."
        return "\n".join(lines[:5])

    def _summarize_with_llm(
        self,
        query: str,
        word_matches: Sequence[StandardWord],
        term_matches: Sequence[StandardTerm],
        vector_matches: Sequence[SimilarTermResult],
        web_matches: Sequence[dict[str, Any]],
    ) -> str | None:
        """LLM이 설정된 경우 검색 근거 기반 요약을 생성합니다."""

        if self.services.llm is None:
            return None
        context = {
            "standard_words": [model_to_dict(item) for item in word_matches],
            "standard_terms": [model_to_dict(item) for item in term_matches],
            "vector_matches": [model_to_dict(item) for item in vector_matches],
            "web_matches": web_matches[:3],
        }
        messages = [
            SystemMessage(
                content=(
                    "너는 표준용어 문의 응답 Agent다. 제공된 RDB, Vector DB, 웹 검색 근거만 사용해 "
                    "한국어로 간결히 답한다. 내부 추론 과정은 출력하지 않는다."
                )
            ),
            HumanMessage(content=f"질문: {query}\n근거 JSON:\n{to_json(context)}"),
        ]
        try:
            raw = self.services.llm.invoke(messages)
            content = str(getattr(raw, "content", "")).strip()
            return content or None
        except Exception as exc:
            LOGGER.info("LLM 문의 요약 실패, deterministic fallback 사용: %s", exc)
            return None


# -----------------------------------------------------------------------------
# SuperAgent입니다.
# 하위 Agent를 LangGraph 노드로 등록하고, intent에 따라 edge를 분기합니다.
# -----------------------------------------------------------------------------
class SuperAgent:
    """LangGraph를 이용해 여러 Sub Agent를 orchestration하는 상위 Agent입니다."""

    def __init__(self, services: RuntimeServices) -> None:
        self.services = services
        self.router_agent = RouterAgent(services)
        self.select_agent = SelectAgent(services)
        self.recommend_agent = RecommendAgent(services)
        self.validate_agent = ValidateAgent(services)
        self.change_agent = ChangeAgent(services)
        self.request_agent = RequestAgent(services)
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """LangGraph 노드/엣지 구조를 명시적으로 구성합니다."""

        if StateGraph is None:
            LOGGER.warning("LangGraph를 import할 수 없어 manual fallback runner를 사용합니다: %s", LANGGRAPH_IMPORT_ERROR)
            return ManualGraphRunner(self)

        graph = StateGraph(AgentState)
        graph.add_node("router_agent", self.router_agent.run)
        graph.add_node("select_agent", self.select_agent.run)
        graph.add_node("recommend_agent", self.recommend_agent.run)
        graph.add_node("validate_agent", self.validate_agent.run)
        graph.add_node("change_agent", self.change_agent.run)
        graph.add_node("request_agent", self.request_agent.run)

        graph.add_edge(START, "router_agent")
        graph.add_conditional_edges(
            "router_agent",
            self._route_by_intent,
            {
                "meaning_search": "select_agent",
                "term_recommend": "recommend_agent",
                "term_register": "validate_agent",
                "term_change": "validate_agent",
                "request": "request_agent",
            },
        )
        graph.add_edge("select_agent", END)
        graph.add_edge("recommend_agent", END)
        graph.add_edge("request_agent", END)
        graph.add_conditional_edges(
            "validate_agent",
            self._route_after_validation,
            {
                "change_agent": "change_agent",
                "end": END,
            },
        )
        graph.add_edge("change_agent", END)
        if MemorySaver is None:
            return graph.compile()
        return graph.compile(checkpointer=MemorySaver())

    def _route_by_intent(self, state: AgentState) -> str:
        """router_agent의 JSON 분류 결과에 따라 다음 노드를 선택합니다."""

        return state.get("intent", "request")

    def _route_after_validation(self, state: AgentState) -> str:
        """검증 성공 및 확인 여부에 따라 change_agent 실행 여부를 결정합니다."""

        validation = state.get("validation") or {}
        if not validation.get("ok"):
            return "end"
        if state.get("intent") == "term_register":
            return "change_agent"
        if state.get("intent") == "term_change" and state.get("apply_change"):
            return "change_agent"
        return "end"

    def invoke(self, state: AgentState, thread_id: str) -> AgentState:
        """대화 thread_id를 기준으로 LangGraph를 실행합니다."""

        runtime_config = {"configurable": {"thread_id": thread_id}}
        state.setdefault("turn_id", str(uuid4()))
        if "forced_intent" not in state:
            state["forced_intent"] = cast(IntentName, "")
        if "apply_change" not in state:
            state["apply_change"] = False
        return cast(AgentState, self.graph.invoke(state, config=runtime_config))


def last_user_message(state: AgentState) -> str:
    """state.messages에서 마지막 HumanMessage 내용을 반환합니다."""

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return state.get("user_input", "")


class ManualGraphRunner:
    """LangGraph 미설치 환경에서 mock smoke test를 가능하게 하는 수동 runner입니다.

    운영/개발 의존성이 설치된 환경에서는 SuperAgent._build_graph가 실제 StateGraph를 반환합니다.
    이 fallback은 동일한 router -> sub agent -> validate -> change 순서를 따라 실행하므로,
    의존성 설치 전에도 데이터/검증 로직을 확인할 수 있습니다.
    """

    def __init__(self, super_agent: SuperAgent) -> None:
        self.super_agent = super_agent
        self.memory: dict[str, AgentState] = {}

    def invoke(self, state: AgentState, config: dict[str, Any] | None = None) -> AgentState:
        """LangGraph compiled graph의 invoke와 같은 형태로 state를 처리합니다."""

        thread_id = "manual"
        if config:
            thread_id = str(config.get("configurable", {}).get("thread_id", thread_id))
        current = merge_agent_state(self.memory.get(thread_id, {}), state)
        current = merge_agent_state(current, self.super_agent.router_agent.run(current))

        intent = current.get("intent", "request")
        if intent == "meaning_search":
            current = merge_agent_state(current, self.super_agent.select_agent.run(current))
        elif intent == "term_recommend":
            current = merge_agent_state(current, self.super_agent.recommend_agent.run(current))
        elif intent in {"term_register", "term_change"}:
            current = merge_agent_state(current, self.super_agent.validate_agent.run(current))
            validation = current.get("validation") or {}
            should_change = bool(validation.get("ok")) and (
                intent == "term_register" or (intent == "term_change" and current.get("apply_change"))
            )
            if should_change:
                current = merge_agent_state(current, self.super_agent.change_agent.run(current))
        else:
            current = merge_agent_state(current, self.super_agent.request_agent.run(current))

        self.memory[thread_id] = current
        return current


def merge_agent_state(left: AgentState, right: AgentState) -> AgentState:
    """ManualGraphRunner에서 LangGraph reducer와 유사하게 state를 병합합니다."""

    merged: AgentState = dict(left)
    for key, value in right.items():
        if key == "messages":
            merged["messages"] = add_messages(merged.get("messages", []), cast(list[BaseMessage], value))
        elif key == "logs":
            merged["logs"] = append_list(merged.get("logs", []), cast(list[dict[str, Any]], value))
        else:
            merged[key] = value  # type: ignore[literal-required]
    return merged


# -----------------------------------------------------------------------------
# Streamlit UI입니다.
# 사용자 입력, Agent 응답, JSON intent 분류 결과, Agent 처리 로그를 한 화면에서 확인할 수 있습니다.
# -----------------------------------------------------------------------------
def _build_super_agent() -> SuperAgent:
    """Streamlit cache_resource로 감쌀 SuperAgent 생성 함수입니다."""

    return SuperAgent(create_runtime_services())


if st is not None:
    get_super_agent = st.cache_resource(show_spinner=False)(_build_super_agent)
else:
    get_super_agent = _build_super_agent


def render_streamlit_app() -> None:
    """Streamlit 화면을 렌더링합니다."""

    if st is None:
        raise RuntimeError("Streamlit이 설치되어 있지 않습니다.")

    st.set_page_config(page_title="표준용어 Agent", layout="wide")
    st.title("표준용어 Agent")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"standard-term-{uuid4()}"
    if "last_state" not in st.session_state:
        st.session_state.last_state = {}

    user_input = st.chat_input("용어 설명 또는 요청을 입력하세요")
    if user_input:
        invoke_from_streamlit({"user_input": user_input, "messages": [HumanMessage(content=user_input)]})

    last_state = cast(AgentState, st.session_state.last_state or {})
    render_messages(last_state)
    render_action_buttons(last_state)
    render_agent_logs(last_state)


def invoke_from_streamlit(state: AgentState) -> None:
    """Streamlit 세션의 thread_id로 SuperAgent를 실행합니다."""

    agent = get_super_agent()
    result = agent.invoke(state, thread_id=st.session_state.thread_id)
    st.session_state.last_state = result


def render_messages(state: AgentState) -> None:
    """LangGraph messages를 chat UI로 표시합니다."""

    for message in state.get("messages", []):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(str(message.content))


def render_action_buttons(state: AgentState) -> None:
    """추천, 신청, 변경 확인 버튼을 표시합니다."""

    columns = st.columns([1, 1, 1])
    with columns[0]:
        if int(state.get("search_count", 0)) >= 1 and st.button("추천", use_container_width=True):
            description = state.get("last_description", "")
            invoke_from_streamlit(
                {
                    "user_input": "추천",
                    "forced_intent": "term_recommend",
                    "active_description": description,
                    "messages": [HumanMessage(content="추천")],
                }
            )
            st.rerun()

    recommendations = state.get("recommendations") or []
    selected_term = ""
    if recommendations:
        names = [str(item.get("logical_name")) for item in recommendations]
        selected_term = st.radio("추천 용어", names, horizontal=True)
    with columns[1]:
        if selected_term and st.button("신청", use_container_width=True):
            invoke_from_streamlit(
                {
                    "user_input": f"{selected_term} 신청",
                    "forced_intent": "term_register",
                    "selected_term": selected_term,
                    "active_description": state.get("active_description") or state.get("last_description", ""),
                    "messages": [HumanMessage(content=f"{selected_term} 신청")],
                }
            )
            st.rerun()

    validation = state.get("validation") or {}
    pending_change = state.get("pending_change_request") or {}
    with columns[2]:
        if validation.get("ok") and pending_change and st.button("변경", use_container_width=True):
            invoke_from_streamlit(
                {
                    "user_input": "변경 실행",
                    "forced_intent": "term_change",
                    "pending_change_request": pending_change,
                    "apply_change": True,
                    "messages": [HumanMessage(content="변경 실행")],
                }
            )
            st.rerun()


def render_agent_logs(state: AgentState) -> None:
    """의도 분류 JSON과 Agent 처리 로그를 표시합니다."""

    with st.expander("Agent 처리 로그", expanded=True):
        if state.get("classification_json"):
            st.caption("router_agent JSON")
            st.code(state["classification_json"], language="json")
        logs = state.get("logs", [])
        if logs:
            st.caption("trace events")
            st.code(to_json(logs[-50:]), language="json")
        else:
            st.write("아직 처리 로그가 없습니다.")


# -----------------------------------------------------------------------------
# CLI 실행 지원입니다.
# Streamlit이 아닌 `python 05.05_prompt_v1.py "작업이 시작된 일시"` 형태로도 mock 동작을 확인할 수 있습니다.
# -----------------------------------------------------------------------------
def is_streamlit_runtime() -> bool:
    """현재 파일이 Streamlit runtime에서 실행 중인지 확인합니다."""

    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_cli() -> None:
    """간단한 CLI 테스트 진입점입니다."""

    if len(sys.argv) <= 1:
        print_execution_guide()
        return
    user_input = " ".join(sys.argv[1:])
    agent = _build_super_agent()
    result = agent.invoke(
        {"user_input": user_input, "messages": [HumanMessage(content=user_input)]},
        thread_id=f"cli-{uuid4()}",
    )
    print(result.get("response", "응답이 없습니다."))
    if result.get("classification_json"):
        print("\n[router_agent JSON]")
        print(result["classification_json"])


def print_execution_guide() -> None:
    """터미널 실행 시 필요한 실행 방법을 출력합니다."""

    guide = f"""
실행 방법
1. Python 3.14+ 환경을 준비합니다.
2. 필요한 패키지를 설치합니다.
   pip install streamlit langgraph langchain langchain-openai langchain-community pydantic python-dotenv sqlalchemy oracledb faiss-cpu tavily-python langsmith opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
3. Mock 실행은 .env에 USE_MOCK_DATA=true를 설정합니다.
4. 실제 OpenAI/Oracle 실행은 .env에 아래 값을 설정합니다.
   OPENAI_API_KEY=...
   OPENAI_MODEL=gpt-5
   EMBEDDING_MODEL=text-embedding-3-large
   USE_MOCK_DATA=false
   ORACLE_DATABASE_URL=oracle+oracledb://user:password@host:1521/?service_name=service
   LANGSMITH_API_KEY=...
   LANGSMITH_PROJECT=standard-term-agent
   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
   TAVILY_API_KEY=...
5. Streamlit UI를 실행합니다.
   streamlit run "{BASE_DIR / '05.05_prompt_v1.py'}"
6. CLI smoke test도 가능합니다.
   python "{BASE_DIR / '05.05_prompt_v1.py'}" "작업이 시작된 일시"
"""
    print(guide.strip())


if __name__ == "__main__":
    if is_streamlit_runtime():
        render_streamlit_app()
    else:
        run_cli()


# -----------------------------------------------------------------------------
# 실행 방법 요약
# - Mock 데이터: .env에 USE_MOCK_DATA=true 작성 후 `streamlit run 05.05_prompt_v1.py`
# - 실제 RDB: USE_MOCK_DATA=false 및 ORACLE_DATABASE_URL 또는 ORACLE_* 환경변수 설정
# - 추적: LANGSMITH_API_KEY가 있으면 LangSmith trace 활성화, OTEL_EXPORTER_OTLP_ENDPOINT가 있으면 OTel 전송
# - CLI 확인: `python 05.05_prompt_v1.py "작업이 시작된 일시"`
# -----------------------------------------------------------------------------
