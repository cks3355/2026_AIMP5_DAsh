# DAsh Technical Requirements

관련 문서: [PRD](prd.md), [Agents](agents.md), [Skill Guide](skill.md)

## 런타임 구조

DAsh는 단일 Python 파일로 실행되는 Streamlit/CLI 겸용 애플리케이션이다. 제품 범위는 [PRD](prd.md), Agent 책임은 [Agents](agents.md), skill 규칙은 [Skill Guide](skill.md)를 기준으로 한다.

`build_runtime()`은 환경설정, 저장소, embedding, vector repository, graph, super agent, 프로젝트 문서를 조립한다.

## 주요 컴포넌트

- AppConfig: `.env`와 환경변수에서 실행 모드, API 키, DB URL, 모델명을 읽는다.
- TermRepository: 표준용어와 표준단어 RDB 접근 인터페이스다.
- VectorRepository: 유사도 검색과 embedding 저장소 접근 인터페이스다.
- ToolBox: Sub Agent가 호출하는 tool registry다.
- DashGraph: LangGraph node/edge 또는 fallback 흐름으로 Sub Agent를 실행한다.
- PiMonoSuperAgent: routing, memory, event stream, graph 실행을 총괄한다.
- ProjectDocuments: v6에서 추가된 프로젝트 명세 문서 로더다.

## 데이터 모델

- StandardTerm: logical_name, description, physical_name, domain, updated_at.
- StandardWord: logical_name, physical_name, meaning_hint.
- SimilarTerm: StandardTerm과 similarity.
- Recommendation: logical_name, description, reason, score.
- ValidationResult: ok, reason, normalized_term.
- ChangeRequest: action, logical_name, description, new_logical_name.
- AgentTurnState: 한 turn 동안의 메시지, 메모리, routing, 조회 결과, 추천 결과, 검증 결과, 최종 응답, 로그를 포함한다.

## 저장소 요구사항

- MockTermRepository는 개발과 데모를 위한 메모리 RDB 역할을 한다.
- OracleTermRepository는 SQLAlchemy를 통해 standard_terms, standard_words 테이블에 접근한다.
- MockVectorRepository는 deterministic hash embedding으로 cosine 검색을 수행한다.
- FaissVectorRepository는 운영 Vector DB 교체 지점으로 유지한다.

## Graph 요구사항

- START에서 router decision에 따라 다음 Agent를 선택한다. Agent별 책임은 [Agents의 Agent 구성](agents.md#agent-구성)을 따른다.
- validate_agent 이후에는 검증 성공 시에만 change_agent로 이동한다.
- 모든 처리 경로는 finalize_agent를 거쳐 END로 종료한다.
- LangGraph가 설치되지 않은 환경에서는 동일한 흐름을 fallback Python 분기로 실행한다.

## UI 요구사항

- Streamlit wide layout을 사용한다.
- 입력 영역과 조회/추천/변경 버튼을 상단에 배치한다.
- 조회 결과, 추천 결과, 변경 결과를 3열로 표시한다.
- Agent 로그는 별도 오른쪽 영역에 표시한다.
- v6 문서 요약은 사이드바 expander로 제공한다.

## 검증 요구사항

- Python 문법 검사는 바이트코드 생성을 피하는 `compile()` 방식으로 수행할 수 있어야 한다.
- Mock 모드 CLI 시나리오가 외부 네트워크 없이 실행되어야 한다.
- 변경 반영은 [Skill Guide의 ValidateSkill](skill.md#validateskill) 승인 결과를 전제로 해야 한다.
