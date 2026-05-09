# DAsh Agents

관련 문서: [PRD](prd.md), [TRD](trd.md), [Skill Guide](skill.md)

## 목적

DAsh의 Agent 책임과 협업 경계를 정의한다. 제품 목표와 사용자 흐름은 [PRD](prd.md)를, 런타임과 graph 구현은 [TRD](trd.md)를 기준으로 한다.

## Agent 구성

### Super Agent

- 사용자 입력을 하나의 turn으로 수신한다.
- RouterSkill을 호출해 intent JSON을 생성한다.
- 메시지, 메모리, lookup_count, SSE 이벤트를 관리한다.
- Sub Agent graph 실행 결과를 최종 응답과 이벤트 로그로 반환한다.

### select_agent

- meaning_select, da_request intent를 처리한다.
- 조회 도구 구성과 저장소 세부사항은 [TRD의 저장소 요구사항](trd.md#저장소-요구사항)을 따른다.
- 모호한 입력은 변경이나 추천으로 넘기지 않고 명확화 질문을 우선한다.

### recommend_agent

- RDB 표준단어를 조회한다.
- 추천 로직은 [Skill Guide의 RecommendSkill](skill.md#recommendskill)을 따른다.

### validate_agent

- 등록, 수정, 삭제 요청을 변경 전 검증한다.
- 검증 규칙은 [Skill Guide의 ValidateSkill](skill.md#validateskill)을 따른다.
- 검증 실패 시 change_agent로 진행하지 않는다.

### change_agent

- validate_agent가 승인한 요청만 처리한다.
- RDB 변경 후 Vector DB를 순차 반영한다.
- 실패 시 사용자에게 취소 또는 오류 메시지를 반환한다.

### finalize_agent

- 오류, 명확화 질문, 검증 실패, 정상 결과를 최종 응답으로 정리한다.
- assistant 메시지와 agent_end 이벤트를 추가한다.

## 이벤트 원칙

- turn_start, message_start, message_end, routing_end, tool_execution_end, turn_end, agent_end 이벤트를 남긴다.
- UI 표시 방식은 [TRD의 UI 요구사항](trd.md#ui-요구사항)을 따른다.
- 이벤트 payload는 디버깅 가능한 최소 구조를 유지하되 민감정보를 포함하지 않는다.

## 운영 원칙

- Mock/운영 모드 구성은 [TRD의 런타임 구조](trd.md#런타임-구조)를 따른다.
- Agent 내부 메시지는 AgentMessage로 유지하고, provider 호출 경계에서만 LLM 메시지로 변환한다.
