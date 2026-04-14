import os
import re
from typing import Dict, List

import pandas as pd
import streamlit as st

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


STANDARD_TERMS: List[Dict[str, str]] = [
    {
        "standard_word": "고객식별번호",
        "domain": "고객",
        "description": "고객을 유일하게 식별하기 위한 번호",
        "synonyms": "고객번호, 고객ID, 회원번호",
    },
    {
        "standard_word": "주문접수일시",
        "domain": "주문",
        "description": "주문이 접수된 날짜와 시간",
        "synonyms": "주문일시, 주문등록일시",
    },
    {
        "standard_word": "상품코드",
        "domain": "상품",
        "description": "상품을 식별하기 위한 코드",
        "synonyms": "품목코드, 제품코드",
    },
    {
        "standard_word": "배송완료일자",
        "domain": "배송",
        "description": "배송이 완료된 날짜",
        "synonyms": "배송완료일, 수령일자",
    },
    {
        "standard_word": "청구금액",
        "domain": "정산",
        "description": "고객에게 청구되는 금액",
        "synonyms": "과금금액, 청구액",
    },
]

KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "title": "표준용어 관리 가이드",
        "content": (
            "표준용어는 업무 의미가 명확하고 중복이 없어야 합니다. "
            "설명 기반 조회 후 가장 근접한 표준단어를 조합하여 후보 용어를 만들 수 있습니다."
        ),
    },
    {
        "title": "추천용어 작성 원칙",
        "content": (
            "추천용어는 사용자 설명의 핵심 명사와 표준단어를 결합해 만듭니다. "
            "너무 긴 문장은 제외하고 식별자, 일자, 금액 같은 표준 접미어를 우선 사용합니다."
        ),
    },
    {
        "title": "문의 응답 정책",
        "content": (
            "사용자가 질문이나 문의를 입력하면 표준용어 조회 대신 문서 검색을 수행하고, "
            "검색 결과를 근거로 LLM이 최종 답변을 생성합니다."
        ),
    },
]

QUESTION_HINTS = [
    "문의",
    "질문",
    "어떻게",
    "왜",
    "언제",
    "가능",
    "방법",
    "알려",
    "설명해",
    "무엇",
    "뭐",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
    return [token for token in cleaned.lower().split() if len(token) >= 2]


def is_inquiry(text: str) -> bool:
    normalized = normalize_text(text)
    if "?" in text:
        return True
    return any(hint in normalized for hint in QUESTION_HINTS)


def search_standard_terms(user_input: str) -> List[Dict[str, str]]:
    query_tokens = set(tokenize(user_input))
    results = []

    for term in STANDARD_TERMS:
        target_text = " ".join(
            [
                term["standard_word"],
                term["domain"],
                term["description"],
                term["synonyms"],
            ]
        )
        term_tokens = set(tokenize(target_text))
        overlap = query_tokens & term_tokens
        score = len(overlap)

        for raw_piece in [term["standard_word"], term["description"], term["synonyms"]]:
            if normalize_text(raw_piece) in normalize_text(user_input):
                score += 2

        if score > 0:
            results.append(
                {
                    "표준단어": term["standard_word"],
                    "도메인": term["domain"],
                    "설명": term["description"],
                    "유사어": term["synonyms"],
                    "점수": score,
                }
            )

    results.sort(key=lambda item: item["점수"], reverse=True)
    return results


def build_term_recommendations(user_input: str, standard_results: List[Dict[str, str]]) -> List[str]:
    if not standard_results:
        return []

    top_word = standard_results[0]["표준단어"]
    tokens = tokenize(user_input)
    candidate_prefixes = [token for token in tokens if token not in tokenize(top_word)]

    if not candidate_prefixes:
        candidate_prefixes = ["업무", "사용자", "대상"]

    recommendations = []
    for prefix in candidate_prefixes[:3]:
        candidate = top_word if prefix in top_word else f"{prefix}{top_word}"
        if candidate not in recommendations:
            recommendations.append(candidate)

    if top_word not in recommendations:
        recommendations.append(top_word)

    return recommendations[:4]


def retrieve_documents(query: str, top_k: int = 2) -> List[Dict[str, str]]:
    query_tokens = set(tokenize(query))
    scored_docs = []

    for doc in KNOWLEDGE_BASE:
        doc_tokens = set(tokenize(doc["title"] + " " + doc["content"]))
        score = len(query_tokens & doc_tokens)
        if score > 0:
            scored_docs.append({**doc, "score": score})

    scored_docs.sort(key=lambda item: item["score"], reverse=True)
    return scored_docs[:top_k] if scored_docs else KNOWLEDGE_BASE[:top_k]


def build_fallback_answer(query: str, docs: List[Dict[str, str]]) -> str:
    context = " ".join(doc["content"] for doc in docs)
    return (
        f"문의 내용: {query}\n\n"
        "프로토타입 기준 답변입니다.\n"
        f"관련 문서를 바탕으로 보면 {context}"
    )


def generate_rag_answer(query: str, docs: List[Dict[str, str]]) -> str:
    if not os.getenv("OPENAI_API_KEY") or ChatOpenAI is None:
        return build_fallback_answer(query, docs)

    context_text = "\n\n".join(
        [f"[문서: {doc['title']}]\n{doc['content']}" for doc in docs]
    )

    prompt = f"""
당신은 표준용어 관리 도우미입니다.
아래 검색 문맥만 근거로 사용자의 문의에 한국어로 답변하세요.
문맥에 없는 내용은 추정하지 말고, 필요한 경우 부족한 점을 짧게 밝히세요.

[사용자 문의]
{query}

[검색 문맥]
{context_text}
""".strip()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)
    return response.content


def reset_result_state() -> None:
    st.session_state["mode"] = None
    st.session_state["term_results"] = []
    st.session_state["recommendations"] = []
    st.session_state["rag_docs"] = []
    st.session_state["rag_answer"] = ""


def init_session_state() -> None:
    defaults = {
        "mode": None,
        "last_input": "",
        "term_results": [],
        "recommendations": [],
        "rag_docs": [],
        "rag_answer": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

st.set_page_config(page_title="표준용어 조회 프로토타입", page_icon="📘", layout="wide")

st.title("표준용어 조회 및 문의 응답 프로토타입")
st.caption("설명을 입력하면 표준용어를 조회하고, 문의성 입력이면 RAG 기반 답변을 표시합니다.")

with st.sidebar:
    st.subheader("프로토타입 안내")
    st.write("- 일반 설명 입력: 표준용어 조회")
    st.write('- 조회 결과 존재 시: "용어 추천" 버튼 노출')
    st.write("- 문의/질문 입력: 문서 검색 후 LLM 답변")
    st.write("- `OPENAI_API_KEY`가 없으면 내장 fallback 답변 사용")

user_input = st.text_area(
    "설명 또는 문의를 입력하세요",
    value=st.session_state["last_input"],
    height=160,
    placeholder="예: 고객을 구분하기 위한 번호\n예: 표준용어 추천은 어떤 기준으로 하나요?",
)

col1, col2 = st.columns([1, 1])

with col1:
    search_clicked = st.button("조회 실행", type="primary", use_container_width=True)

with col2:
    clear_clicked = st.button("초기화", use_container_width=True)

if clear_clicked:
    st.session_state["last_input"] = ""
    reset_result_state()
    st.rerun()

if search_clicked:
    clean_input = user_input.strip()
    st.session_state["last_input"] = clean_input
    reset_result_state()

    if not clean_input:
        st.warning("설명 또는 문의 내용을 입력해주세요.")
    elif is_inquiry(clean_input):
        docs = retrieve_documents(clean_input)
        answer = generate_rag_answer(clean_input, docs)
        st.session_state["mode"] = "rag"
        st.session_state["rag_docs"] = docs
        st.session_state["rag_answer"] = answer
    else:
        results = search_standard_terms(clean_input)
        st.session_state["mode"] = "term"
        st.session_state["term_results"] = results

if st.session_state["mode"] == "term":
    st.subheader("표준용어 조회 결과")
    term_results = st.session_state["term_results"]

    if term_results:
        df = pd.DataFrame(term_results)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("용어 추천", use_container_width=False):
            st.session_state["recommendations"] = build_term_recommendations(
                st.session_state["last_input"],
                term_results,
            )

        if st.session_state["recommendations"]:
            st.subheader("추천용어")
            for idx, item in enumerate(st.session_state["recommendations"], start=1):
                st.write(f"{idx}. {item}")
    else:
        st.info("입력 설명과 매칭되는 표준용어를 찾지 못했습니다.")

if st.session_state["mode"] == "rag":
    st.subheader("문의 답변")
    st.write(st.session_state["rag_answer"])

    with st.expander("검색에 사용된 문서 보기"):
        for doc in st.session_state["rag_docs"]:
            st.markdown(f"**{doc['title']}**")
            st.write(doc["content"])



