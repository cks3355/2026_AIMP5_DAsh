# LangGraph + Streamlit 기반 표준용어 Multi Agent 챗봇 백본입니다.
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


# 앱에서 사용할 파일 경로와 유사도 기준값을 정의합니다.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "standard_terms.db"
SIMILARITY_THRESHOLD = 0.55


# SQLite 연결을 생성합니다. 일부 Windows 샌드박스에서 WAL 잠금이 막혀 journal을 끕니다.
def connect_rdb() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=OFF")
    return conn


# LangGraph 노드들이 공유하는 상태 스키마입니다.
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    query: str
    term_results: list[dict[str, Any]]
    recommendations: list[str]
    selected_term: str
    change_request: dict[str, str]
    validation: dict[str, Any]
    response: str


# Super Agent가 선택할 수 있는 Sub Agent 명세입니다.
@dataclass(frozen=True)
class SubAgent:
    name: str
    description: str


SUB_AGENTS = {
    "search": SubAgent("search", "표준용어 설명을 기반으로 Vector DB에서 유사 용어를 조회"),
    "recommend": SubAgent("recommend", "표준단어 조합으로 신규 표준용어 후보를 추천"),
    "register": SubAgent("register", "추천 용어를 검증한 뒤 RDB와 Vector DB에 등록"),
    "change": SubAgent("change", "표준용어 수정/삭제 가능 여부를 검증하고 변경"),
    "qa": SubAgent("qa", "RDB, Vector DB, 웹 지식 요약 형태로 문의 답변"),
}


# 데모용 표준단어와 속성분류어를 정의합니다.
STANDARD_WORDS = {
    "작업": {"kind": "business", "synonyms": ["작업", "일", "업무"]},
    "시작": {"kind": "business", "synonyms": ["시작", "개시", "start"]},
    "종료": {"kind": "business", "synonyms": ["종료", "완료", "끝"]},
    "일시": {"kind": "class", "synonyms": ["일시", "시점", "시간", "때"]},
    "재공": {"kind": "business", "synonyms": ["재공", "공정중", "미완성"]},
    "수량": {"kind": "class", "synonyms": ["수량", "개수", "량"]},
}


# SQLite RDB를 초기화하고 기본 표준용어를 적재합니다.
def init_rdb() -> None:
    with connect_rdb() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS standard_terms (
                logical_name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                domain TEXT DEFAULT '공통',
                status TEXT DEFAULT 'ACTIVE'
            )
            """
        )
        seed_terms = [
            ("작업시작일시", "작업이 start된 시점", "생산"),
            ("재공수량", "제조 공정에서 생산 중이거나 가공 중인 미완성 제품의 수량", "제조"),
        ]
        conn.executemany(
            """
            INSERT OR IGNORE INTO standard_terms(logical_name, description, domain)
            VALUES (?, ?, ?)
            """,
            seed_terms,
        )


# RDB에서 활성 표준용어 목록을 읽습니다.
def load_terms() -> list[dict[str, str]]:
    init_rdb()
    with connect_rdb() as conn:
        rows = conn.execute(
            """
            SELECT logical_name, description, domain
            FROM standard_terms
            WHERE status = 'ACTIVE'
            ORDER BY logical_name
            """
        ).fetchall()
    return [{"logical_name": name, "description": desc, "domain": domain} for name, desc, domain in rows]


# 간결한 로컬 Vector DB 역할을 하는 코사인 유사도 검색기입니다.
class SimpleVectorDB:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.vectors = [self._embed(f"{row['logical_name']} {row['description']}") for row in rows]

    def _embed(self, text: str) -> Counter:
        normalized = text.lower().replace("start", "시작")
        tokens = re.findall(r"[가-힣a-zA-Z0-9]+", normalized)
        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            for word, meta in STANDARD_WORDS.items():
                if token in word or word in token or token in meta["synonyms"]:
                    expanded.append(word)
        return Counter(expanded)

    def _cosine(self, left: Counter, right: Counter) -> float:
        common = set(left) & set(right)
        dot = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        query_vector = self._embed(query)
        scored = []
        for row, vector in zip(self.rows, self.vectors):
            scored.append({**row, "score": round(self._cosine(query_vector, vector), 4)})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:k]


# RDB 기준으로 Vector DB를 재생성합니다.
def get_vector_db() -> SimpleVectorDB:
    return SimpleVectorDB(load_terms())


# 마지막 사용자 메시지를 추출합니다.
def last_user_message(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return state.get("query", "")


# Super Agent가 사용자 의도를 분류해 다음 Sub Agent를 결정합니다.
def super_agent(state: AgentState) -> dict[str, str]:
    query = last_user_message(state).strip()
    compact = query.replace(" ", "")

    if state.get("intent"):
        return {"intent": state["intent"], "query": query}
    if any(word in compact for word in ["변경", "수정", "삭제"]):
        return {"intent": "change", "query": query}
    if "추천" in compact:
        return {"intent": "recommend", "query": query}
    if compact in STANDARD_WORDS or any(word in compact for word in ["무엇", "뭐", "란?", "있나요", "문의", "체크리스트"]):
        return {"intent": "qa", "query": query}
    return {"intent": "search", "query": query}


# 표준용어 검색 Sub Agent입니다.
def search_agent(state: AgentState) -> dict[str, Any]:
    results = get_vector_db().search(state["query"])
    lines = [
        f"{item['logical_name']}: {item['description']} (유사도 {item['score']})"
        for item in results
        if item["score"] >= SIMILARITY_THRESHOLD
    ]
    response = "\n".join(lines) if lines else "유사한 표준용어를 찾지 못했습니다. 추천을 요청할 수 있습니다."
    return {"term_results": results, "response": response, "messages": [AIMessage(content=response)]}


# 표준단어 논리명만 사용해 신규 표준용어를 추천하는 Sub Agent입니다.
def recommend_agent(state: AgentState) -> dict[str, Any]:
    query = state["query"]
    selected_words = []
    for word, meta in STANDARD_WORDS.items():
        if any(alias in query for alias in [word, *meta["synonyms"]]):
            selected_words.append(word)

    if "일시" not in selected_words and any(token in query for token in ["일시", "시점", "시간", "때"]):
        selected_words.append("일시")
    if selected_words and STANDARD_WORDS[selected_words[-1]]["kind"] != "class":
        selected_words.append("일시")

    recommendation = "".join(dict.fromkeys(selected_words)) or "작업종료일시"
    if recommendation in {term["logical_name"] for term in load_terms()}:
        recommendation = recommendation.replace("시작", "종료") if "시작" in recommendation else f"신규{recommendation}"

    response = recommendation
    return {"recommendations": [recommendation], "response": response, "messages": [AIMessage(content=response)]}


# 등록 전 표준 체크리스트를 검증합니다.
def validate_new_term(logical_name: str, description: str) -> dict[str, Any]:
    known_words = [word for word in STANDARD_WORDS if word in logical_name]
    last_word = known_words[-1] if known_words else ""
    errors = []
    if logical_name in {term["logical_name"] for term in load_terms()}:
        errors.append("이미 등록된 표준용어입니다.")
    if not known_words or "".join(known_words) != logical_name:
        errors.append("표준단어 논리명만 조합해야 합니다.")
    if not last_word or STANDARD_WORDS[last_word]["kind"] != "class":
        errors.append("마지막 단어는 속성분류어여야 합니다.")
    if not description.strip():
        errors.append("설명은 필수입니다.")
    return {"ok": not errors, "errors": errors}


# 추천 용어를 RDB에 등록하고 Vector DB 현행화 대상이 되도록 저장합니다.
def register_agent(state: AgentState) -> dict[str, Any]:
    logical_name = state.get("selected_term") or (state.get("recommendations") or [""])[0]
    description = state.get("query", "").strip() or f"{logical_name}에 대한 설명"
    validation = validate_new_term(logical_name, description)

    if not validation["ok"]:
        response = "등록이 불가합니다. " + " ".join(validation["errors"])
        return {"validation": validation, "response": response, "messages": [AIMessage(content=response)]}

    with connect_rdb() as conn:
        conn.execute(
            """
            INSERT INTO standard_terms(logical_name, description, domain)
            VALUES (?, ?, ?)
            """,
            (logical_name, description, "신규"),
        )
    response = "신청이 완료되었습니다. RDB와 Vector DB에 반영되었습니다."
    return {"validation": validation, "response": response, "messages": [AIMessage(content=response)]}


# 변경 요청 문장에서 대상 용어와 변경 내용을 추출합니다.
def parse_change_request(query: str) -> dict[str, str]:
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
    target = quoted[0] if quoted else ""
    new_desc = quoted[1] if len(quoted) > 1 else ""
    action = "delete" if "삭제" in query else "update"
    if not target:
        for term in load_terms():
            if term["logical_name"] in query:
                target = term["logical_name"]
                break
    return {"action": action, "target": target, "description": new_desc}


# 변경 가능 여부를 표준 체크리스트 관점으로 검증합니다.
def validate_change(request: dict[str, str]) -> dict[str, Any]:
    terms = {term["logical_name"]: term for term in load_terms()}
    errors = []
    if not request["target"] or request["target"] not in terms:
        errors.append("대상 표준용어가 RDB에 존재하지 않습니다.")
    if request["action"] == "update" and not request["description"]:
        errors.append("수정할 설명을 따옴표로 명확히 입력해야 합니다.")
    return {"ok": not errors, "errors": errors}


# 변경 검증 및 실제 변경을 수행하는 Sub Agent입니다.
def change_agent(state: AgentState) -> dict[str, Any]:
    request = state.get("change_request") or parse_change_request(state["query"])
    validation = validate_change(request)
    apply_change = state.get("intent") == "apply_change"

    if not validation["ok"]:
        response = " ".join(f"{reason} 사유로 변경이 불가합니다." for reason in validation["errors"])
        return {"change_request": request, "validation": validation, "response": response, "messages": [AIMessage(content=response)]}
    if not apply_change:
        response = "변경이 가능합니다."
        return {"change_request": request, "validation": validation, "response": response, "messages": [AIMessage(content=response)]}

    with connect_rdb() as conn:
        if request["action"] == "delete":
            conn.execute("UPDATE standard_terms SET status = 'DELETED' WHERE logical_name = ?", (request["target"],))
        else:
            conn.execute(
                "UPDATE standard_terms SET description = ? WHERE logical_name = ?",
                (request["description"], request["target"]),
            )
    response = "변경이 완료되었습니다. RDB와 Vector DB에 반영되었습니다."
    return {"change_request": request, "validation": validation, "response": response, "messages": [AIMessage(content=response)]}


# 문의 답변 Sub Agent입니다. 웹 검색은 실제 연동 지점으로 분리하고 여기서는 요약 형식을 유지합니다.
def qa_agent(state: AgentState) -> dict[str, Any]:
    query = state["query"]
    rdb_hits = [term for term in load_terms() if term["logical_name"] in query or term["description"] in query]
    vector_hits = get_vector_db().search(query, k=2)

    if "재공" in query:
        web_summary = "제조 공정에서 생산 중이거나 가공 중인 미완성 제품을 의미합니다."
    elif "체크리스트" in query:
        web_summary = "표준 체크리스트는 용어 중복, 표준단어 조합, 속성분류어 사용, 설명 명확성을 기준으로 확인합니다."
    else:
        web_summary = "외부 웹 검색 연동 시 관련 문서를 요약해 표시하는 영역입니다."

    rdb_text = "\n".join(f"- {item['logical_name']}: {item['description']}" for item in rdb_hits) or "- 직접 일치 항목 없음"
    vector_text = "\n".join(f"- {item['logical_name']}: {item['description']}" for item in vector_hits) or "- 유사 항목 없음"
    response = f"[RDB]\n{rdb_text}\n\n[Vector DB]\n{vector_text}\n\n[Web 요약]\n- {web_summary}"
    return {"term_results": vector_hits, "response": response, "messages": [AIMessage(content=response)]}


# Super Agent의 라우팅 결과를 LangGraph 조건부 엣지에 전달합니다.
def route_by_intent(state: AgentState) -> str:
    if state["intent"] == "apply_register":
        return "register"
    if state["intent"] == "apply_change":
        return "change"
    return state["intent"]


# Super Agent 1개와 여러 Sub Agent로 구성된 LangGraph를 빌드합니다.
@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("super_agent", super_agent)
    graph.add_node("search", search_agent)
    graph.add_node("recommend", recommend_agent)
    graph.add_node("register", register_agent)
    graph.add_node("change", change_agent)
    graph.add_node("qa", qa_agent)

    graph.add_edge(START, "super_agent")
    graph.add_conditional_edges(
        "super_agent",
        route_by_intent,
        {
            "search": "search",
            "recommend": "recommend",
            "register": "register",
            "change": "change",
            "qa": "qa",
        },
    )
    for agent_name in SUB_AGENTS:
        graph.add_edge(agent_name, END)
    return graph.compile()


# Streamlit 세션 상태를 초기화합니다.
def init_session() -> None:
    init_rdb()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_state", {})


# 그래프를 실행하고 결과를 화면 이력에 저장합니다.
def run_chat(query: str, intent: str | None = None, extra: dict[str, Any] | None = None) -> AgentState:
    app = build_graph()
    state: AgentState = {"messages": [HumanMessage(content=query)]}
    if intent:
        state["intent"] = intent
    if extra:
        state.update(extra)
    result = app.invoke(state)
    st.session_state.last_state = result
    st.session_state.history.append(("user", query))
    st.session_state.history.append(("assistant", result["response"]))
    return result


# 추천/신청 및 변경 버튼을 상황에 맞게 렌더링합니다.
def render_action_buttons() -> None:
    last_state = st.session_state.get("last_state", {})
    query = last_state.get("query", "")

    if last_state.get("term_results") and not last_state.get("recommendations"):
        if st.button("추천", use_container_width=True):
            run_chat(query, intent="recommend")
            st.rerun()

    if last_state.get("recommendations"):
        selected = st.radio("신청할 추천 용어", last_state["recommendations"], horizontal=True)
        if st.button("신청", use_container_width=True):
            run_chat(query, intent="apply_register", extra={"selected_term": selected})
            st.rerun()

    if last_state.get("validation", {}).get("ok") and last_state.get("change_request"):
        if st.button("변경", use_container_width=True):
            run_chat(query, intent="apply_change", extra={"change_request": last_state["change_request"]})
            st.rerun()


# Streamlit 챗봇 UI를 구성합니다.
def main() -> None:
    st.set_page_config(page_title="표준용어 Multi Agent", page_icon="AI", layout="centered")
    init_session()

    st.title("표준용어 Multi Agent 챗봇")
    st.caption("Super Agent가 의도를 분류하고 검색, 추천, 등록, 변경, 문의 Sub Agent를 호출합니다.")

    with st.expander("Agent 구성", expanded=False):
        for agent in SUB_AGENTS.values():
            st.write(f"- {agent.name}: {agent.description}")

    for role, content in st.session_state.history:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("예: 작업이 시작된 일시")
    if user_input:
        run_chat(user_input)
        st.rerun()

    render_action_buttons()


# Streamlit 실행 진입점입니다.
if __name__ == "__main__":
    main()
