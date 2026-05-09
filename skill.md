# DAsh Skill Guide

관련 문서: [PRD](prd.md), [Agents](agents.md), [TRD](trd.md)

이 문서는 skill의 판단 규칙만 정의한다. 어떤 Agent가 skill을 호출하는지는 [Agents](agents.md), 데이터 모델과 런타임 연결은 [TRD](trd.md)를 따른다.

## RouterSkill

RouterSkill은 사용자 입력을 표준용어 업무 intent로 분류한다.

- 삭제, 제거가 포함되면 term_delete로 분류한다.
- 변경, 수정, 바꿔가 포함되면 term_update로 분류한다.
- 신청, 등록, 추가와 인용된 논리명이 함께 있으면 term_insert로 분류한다.
- 추천 또는 직전 조회 이후의 추천 입력은 term_recommend로 분류한다.
- 질문형 표현, 체크리스트, 가이드 요청은 da_request로 분류한다.
- 너무 짧거나 일반적인 단어만 있는 입력은 meaning_select로 두되 명확화 질문을 요청한다.
- 그 외 설명형 입력은 meaning_select로 분류한다.

## RecommendSkill

RecommendSkill은 표준단어를 조합해 후보 용어를 만든다.

- 설명에 포함된 표준단어 또는 meaning_hint와 연결되는 단어를 우선 선택한다.
- 업무 핵심어가 직접 포함되면 후보 단어 목록에 보강한다.
- 여러 조합 경로를 만들고 점수 차이를 둔다.
- 기존 표준용어와 중복되면 후보명을 보정한다.
- 시간 속성 설명인데 일시가 누락되면 일시를 보강한다.
- 결과가 부족하면 fallback 후보를 만들어 최대 3개를 채운다.

## ValidateSkill

ValidateSkill은 변경 요청의 저장 가능성을 판단한다.

- 삭제는 대상 표준용어가 존재할 때만 허용한다.
- 등록 논리명은 3자 이상이어야 한다.
- 논리명은 한글, 영문, 숫자, 밑줄만 허용한다.
- 설명은 8자 이상이어야 한다.
- 등록 대상이 이미 존재하면 실패한다.
- 수정 대상이 존재하지 않으면 실패한다.
- 성공 시 StandardTerm으로 정규화하고 물리명을 생성 또는 유지한다.

## Skill 사용 원칙

- skill은 deterministic fallback으로 동작해야 한다.
- 외부 LLM을 사용할 때도 동일한 입력과 출력 스키마를 유지해야 한다.
- Agent 간 전달 데이터 모델은 [TRD의 데이터 모델](trd.md#데이터-모델)을 따른다.
- 검증 실패는 변경 작업을 호출하지 않는 제어 신호로 사용한다.
