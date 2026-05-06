# LangGraph 기반 Super Agent / Sub Agent 아키텍처 예제입니다.
import os
from dataclasses import dataclass
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# 환경변수에서 LLM 설정을 불러옵니다.
load_dotenv(override=True)
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5")


# LangGraph에서 공유할 Agent 상태를 정의합니다.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    selected_agent: str
    route_reason: str


# Super Agent가 Sub Agent를 선택할 때 사용할 응답 스키마입니다.
class RouteDecision(BaseModel):
    selected_agent: str = Field(description="Selected sub agent name.")
    reason: str = Field(description="Short routing reason.")


# Sub Agent를 쉽게 추가하기 위한 공통 명세입니다.
@dataclass
class AgentSpec:
    name: str
    description: str
    system_prompt: str


# 공통 LLM 객체를 생성합니다.
def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=temperature,
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )


# 도메인별 Sub Agent 목록입니다. 새 Agent는 이 딕셔너리에만 추가하면 됩니다.
SUB_AGENTS: dict[str, AgentSpec] = {
    "planner": AgentSpec(
        name="planner",
        description="요구사항을 분석하고 실행 계획을 세우는 Agent",
        system_prompt="당신은 Planner Agent입니다. 사용자의 요구사항을 단계별 실행 계획으로 정리하세요.",
    ),
    "researcher": AgentSpec(
        name="researcher",
        description="정보를 조사하고 핵심 근거를 요약하는 Agent",
        system_prompt="당신은 Researcher Agent입니다. 필요한 정보를 조사한 것처럼 핵심 근거와 결론을 요약하세요.",
    ),
    "coder": AgentSpec(
        name="coder",
        description="파이썬 코드 설계와 구현을 담당하는 Agent",
        system_prompt="당신은 Coder Agent입니다. 간결하고 실행 가능한 파이썬 중심 답변을 작성하세요.",
    ),
}


# Super Agent가 사용자 요청에 가장 적합한 Sub Agent를 선택합니다.
def super_agent_router(state: AgentState) -> dict:
    llm = get_llm().with_structured_output(RouteDecision)
    agent_descriptions = "\n".join(
        f"- {agent.name}: {agent.description}" for agent in SUB_AGENTS.values()
    )
    messages = [
        SystemMessage(
            content=(
                "당신은 Super Agent입니다. 사용자 요청을 분석해서 가장 적합한 Sub Agent 하나를 선택하세요.\n"
                f"선택 가능한 Sub Agent 목록:\n{agent_descriptions}"
            )
        ),
        *state["messages"],
    ]
    decision = llm.invoke(messages)

    selected_agent = decision.selected_agent
    if selected_agent not in SUB_AGENTS:
        selected_agent = "planner"

    return {
        "selected_agent": selected_agent,
        "route_reason": decision.reason,
    }


# LangGraph 조건부 라우팅에서 선택된 Sub Agent 이름을 반환합니다.
def route_to_sub_agent(state: AgentState) -> str:
    return state["selected_agent"]


# 선택된 Sub Agent를 실행하고 답변을 상태에 추가합니다.
def run_sub_agent(state: AgentState) -> dict:
    selected_agent = state["selected_agent"]
    sub_agent = SUB_AGENTS[selected_agent]
    llm = get_llm()
    messages = [SystemMessage(content=sub_agent.system_prompt), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


# Super Agent와 Sub Agent 실행 흐름을 LangGraph로 구성합니다.
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("super_agent", super_agent_router)
    graph.add_node("sub_agent", run_sub_agent)

    graph.add_edge(START, "super_agent")
    graph.add_conditional_edges(
        "super_agent",
        route_to_sub_agent,
        {agent_name: "sub_agent" for agent_name in SUB_AGENTS},
    )
    graph.add_edge("sub_agent", END)

    return graph.compile()


# 파일을 직접 실행했을 때 간단한 테스트 요청을 수행합니다.
if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        {
            "messages": [HumanMessage(content="고객 데이터를 분석하는 파이썬 구조를 설계해줘.")],
            "selected_agent": "",
            "route_reason": "",
        }
    )
    print(result["messages"][-1].content)
