# 환경 변수와 기본 의존성을 불러옵니다.
import os
from typing import Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# LLM 설정을 초기화합니다.
load_dotenv(override=True)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5")


# LangGraph에서 공유할 상태를 정의합니다.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    selected_agent: str
    route_reason: str


# Super Agent가 선택할 Sub Agent 형식을 정의합니다.
class RouteDecision(BaseModel):
    selected_agent: str = Field(description="Selected sub agent name.")
    reason: str = Field(description="Short routing reason in Korean.")


# 공통 LLM 생성 함수를 정의합니다.
def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=temperature,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )


# 마지막 사용자 메시지를 안전하게 가져옵니다.
def get_last_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


# 간단한 키워드 규칙으로 1차 라우팅 안정성을 확보합니다.
def keyword_route(user_text: str) -> RouteDecision | None:
    lowered = user_text.lower()

    if any(keyword in lowered for keyword in ["plan", "roadmap", "일정", "계획", "전략"]):
        return RouteDecision(
            selected_agent="planner",
            reason="계획 수립 성격의 요청으로 판단했습니다.",
        )

    if any(keyword in lowered for keyword in ["research", "조사", "비교", "trend", "시장", "리서치"]):
        return RouteDecision(
            selected_agent="researcher",
            reason="조사와 비교 중심 요청으로 판단했습니다.",
        )

    if any(keyword in lowered for keyword in ["code", "python", "streamlit", "langgraph", "구현", "개발"]):
        return RouteDecision(
            selected_agent="coder",
            reason="구현 및 코드 생성 요청으로 판단했습니다.",
        )

    return None


# Super Agent가 가장 적합한 Sub Agent를 선택합니다.
def supervisor_node(state: AgentState) -> dict:
    user_text = get_last_user_message(state["messages"])
    fallback = keyword_route(user_text)
    if fallback:
        return {
            "selected_agent": fallback.selected_agent,
            "route_reason": fallback.reason,
        }

    router = get_llm().with_structured_output(RouteDecision)
    decision = router.invoke(
        [
            SystemMessage(
                content=(
                    "당신은 Super Agent입니다. 사용자 요청을 보고 가장 적합한 Sub Agent 하나만 선택하세요.\n"
                    "선택 가능한 agent는 planner, researcher, coder 입니다.\n"
                    "planner: 목표를 단계별 실행 계획으로 정리\n"
                    "researcher: 개념 설명, 비교, 조사, 요약\n"
                    "coder: 코드 설계, 구현, 디버깅\n"
                    "reason은 한국어 한 문장으로 작성하세요."
                )
            ),
            *state["messages"],
        ]
    )

    selected_agent = decision.selected_agent if decision.selected_agent in {
        "planner",
        "researcher",
        "coder",
    } else "coder"

    return {
        "selected_agent": selected_agent,
        "route_reason": decision.reason,
    }


# Planner Sub Agent가 실행 계획을 만듭니다.
def planner_node(state: AgentState) -> dict:
    response = get_llm().invoke(
        [
            SystemMessage(
                content=(
                    "당신은 Planner Agent입니다. 요청을 단계별 실행 계획으로 정리하세요. "
                    "핵심 목표, 우선순위, 다음 액션이 드러나게 간결하게 답하세요."
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


# Researcher Sub Agent가 설명과 비교를 제공합니다.
def researcher_node(state: AgentState) -> dict:
    response = get_llm().invoke(
        [
            SystemMessage(
                content=(
                    "당신은 Researcher Agent입니다. 사용자의 질문을 개념 설명, 장단점 비교, "
                    "의사결정 포인트 중심으로 간결하게 정리하세요."
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


# Coder Sub Agent가 구현 중심 답변을 제공합니다.
def coder_node(state: AgentState) -> dict:
    response = get_llm().invoke(
        [
            SystemMessage(
                content=(
                    "당신은 Coder Agent입니다. 실무적인 코드 설계와 구현 관점으로 답하세요. "
                    "필요하면 짧은 예시 코드와 함께 설명하세요."
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [AIMessage(content=response.content)]}


# 선택된 Sub Agent로 라우팅할 분기 함수를 정의합니다.
def route_to_sub_agent(state: AgentState) -> str:
    return state["selected_agent"]


# Super Agent 1개와 여러 Sub Agent로 그래프를 구성합니다.
@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("coder", coder_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_to_sub_agent,
        {
            "planner": "planner",
            "researcher": "researcher",
            "coder": "coder",
        },
    )
    graph.add_edge("planner", END)
    graph.add_edge("researcher", END)
    graph.add_edge("coder", END)
    return graph.compile()


# Streamlit 세션 상태를 초기화합니다.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "route_reason" not in st.session_state:
    st.session_state.route_reason = ""
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = ""


# Streamlit 챗봇 화면을 구성합니다.
st.set_page_config(page_title="Multi Agent Chatbot", page_icon="R", layout="centered")
st.title("LangGraph Multi Agent Chatbot")
st.caption("Super Agent 1개가 요청을 분류하고, 적합한 Sub Agent가 응답합니다.")


# 사이드바에 현재 라우팅 정보를 보여줍니다.
with st.sidebar:
    st.subheader("Routing")
    st.write(f"Selected Agent: `{st.session_state.selected_agent or '-'}`")
    st.write(st.session_state.route_reason or "아직 라우팅 전입니다.")
    if st.button("대화 초기화"):
        st.session_state.chat_history = []
        st.session_state.route_reason = ""
        st.session_state.selected_agent = ""
        st.rerun()


# 누적된 대화 기록을 채팅 UI에 렌더링합니다.
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)


# 사용자 입력을 받아 LangGraph를 실행하고 결과를 출력합니다.
if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.chat_history.append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("멀티 에이전트가 답변을 작성하는 중입니다..."):
            app = build_graph()
            result = app.invoke(
                {
                    "messages": [HumanMessage(content=prompt)],
                    "selected_agent": "",
                    "route_reason": "",
                }
            )

            answer = result["messages"][-1].content
            st.session_state.selected_agent = result["selected_agent"]
            st.session_state.route_reason = result["route_reason"]
            st.session_state.chat_history.append(("assistant", answer))
            st.markdown(answer)
