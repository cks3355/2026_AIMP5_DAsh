from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
from types import ModuleType

from dotenv import load_dotenv
from openai import AzureOpenAI
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
VECTOR_DB_SCRIPT = BASE_DIR / "04.21_vector_db_registration.py"

load_dotenv(override=True)


def load_vector_db_module() -> ModuleType:
    """벡터 DB 등록 예제 파일을 동적으로 불러온다."""
    spec = importlib.util.spec_from_file_location(
        "vector_db_registration",
        VECTOR_DB_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("벡터 DB 등록 스크립트를 불러오지 못했습니다.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner="표준용어 벡터 DB를 준비하는 중입니다...")
def get_vectorstore():
    """예제 등록 로직을 재사용해 FAISS 벡터 DB를 생성한다."""
    module = load_vector_db_module()
    documents = module.build_standard_term_documents()
    return module.register_terms_to_vector_db(documents)


def extract_description(page_content: str) -> str:
    """Document 본문에서 설명 텍스트를 추출한다."""
    for line in page_content.splitlines():
        normalized_line = line.strip()
        if normalized_line.startswith("설명:"):
            return normalized_line.replace("설명:", "", 1).strip()
    return page_content.strip()


def search_standard_terms(query: str, top_k: int = 5) -> list[dict[str, str | float]]:
    """입력 설명과 유사한 표준용어를 검색한다."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    formatted_results = []
    for document, distance in results:
        description = document.metadata.get("description") or extract_description(
            document.page_content
        )
        similarity = 1 / (1 + float(distance))
        formatted_results.append(
            {
                "standard_term_name": document.metadata.get(
                    "standard_term_name", "표준용어명 없음"
                ),
                "description": description,
                "distance": float(distance),
                "similarity": similarity,
            }
        )

    formatted_results.sort(key=lambda item: item["distance"])
    return formatted_results


@st.cache_resource
def get_llm_client() -> AzureOpenAI:
    """용어 추천에 사용할 Azure OpenAI 클라이언트를 생성한다."""
    return AzureOpenAI(
        azure_endpoint=os.getenv("LLM_BASE_URL", "https://aitalentlab.skax.co.kr"),
        api_key=os.getenv("LLM_API_KEY"),
        api_version="2024-12-01-preview",
    )


def extract_keywords(description: str) -> list[str]:
    """설명에서 네이버 사전 조회에 사용할 한글 키워드를 추출한다."""
    candidates = re.findall(r"[\uac00-\ud7a3]{2,}", description)
    keywords: list[str] = []
    for word in candidates:
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= 5:
            break
    return keywords


def fetch_naver_dictionary_words(description: str, max_words: int = 12) -> list[str]:
    """네이버 사전 검색 결과에서 조합용 단어 후보를 수집한다."""
    keywords = extract_keywords(description)
    collected_words: list[str] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    }

    for keyword in keywords:
        try:
            response = requests.get(
                "https://ko.dict.naver.com/search.nhn",
                params={"query": keyword},
                headers=headers,
                timeout=5,
            )
            response.raise_for_status()
        except requests.RequestException:
            continue

        matches = re.findall(r'"title":"([^"]+)"', response.text)
        for match in matches:
            cleaned = re.sub(r"[^\uac00-\ud7a3]", "", match)
            if len(cleaned) < 2 or cleaned.endswith("\uac12"):
                continue
            if cleaned not in collected_words:
                collected_words.append(cleaned)
            if len(collected_words) >= max_words:
                return collected_words

    for keyword in keywords:
        if keyword not in collected_words:
            collected_words.append(keyword)
        if len(collected_words) >= max_words:
            break

    return collected_words


def fallback_recommend_terms(description: str, dictionary_words: list[str]) -> list[str]:
    """LLM 호출이 어려울 때 규칙 기반으로 추천 용어를 생성한다."""
    seed_words = dictionary_words or extract_keywords(description) or [
        "\uae30\uc900",
        "\uc815\ubcf4",
        "\uad00\ub9ac",
    ]
    while len(seed_words) < 4:
        for extra in [
            "\uae30\uc900",
            "\ub300\uc0c1",
            "\uc2dd\ubcc4",
            "\uc815\ubcf4",
            "\uad00\ub9ac",
        ]:
            if extra not in seed_words:
                seed_words.append(extra)
            if len(seed_words) >= 4:
                break

    recommendations = []
    patterns = [
        seed_words[:3] + ["\uac12"],
        seed_words[:2] + seed_words[3:4] + ["\uac12"],
        seed_words[1:4] + ["\uac12"],
    ]
    for words in patterns:
        normalized = [normalize_term_unit(word) for word in words if word != "\uac12"]
        normalized = [word for word in normalized if word][:4]
        if len(normalized) < 2:
            normalized = [normalize_term_unit(word) for word in seed_words[:2]]
        recommendations.append("".join(normalized[:4] + ["\uac12"]))
    return recommendations[:3]


def normalize_term_unit(word: str) -> str:
    """추천 용어를 구성하는 각 단어를 2~3글자 한글 단위로 정리한다."""
    cleaned = re.sub(r"[^\uac00-\ud7a3]", "", word)
    if len(cleaned) < 2:
        return ""
    if len(cleaned) > 3:
        cleaned = cleaned[:3]
    return cleaned


def sanitize_recommended_term(term: str) -> str:
    """추천 용어를 공백 없이, 각 단어 2~3글자, 마지막 값 형식으로 정리한다."""
    cleaned = re.sub(r"[^\uac00-\ud7a3\s]", " ", term)
    parts = [part for part in cleaned.split() if part]

    filtered_parts: list[str] = []
    for part in parts:
        if part == "\uac12":
            continue
        normalized = normalize_term_unit(part)
        if normalized and normalized not in filtered_parts:
            filtered_parts.append(normalized)

    if not filtered_parts:
        filtered_parts = ["\ud45c\uc900", "\ucd94\ucc9c"]

    filtered_parts = filtered_parts[:4]
    while len(filtered_parts) < 2:
        filtered_parts.append("\uc815\ubcf4")

    if len(filtered_parts) < 3:
        for extra in ["\uae30\uc900", "\uc2dd\ubcc4", "\uad00\ub9ac"]:
            normalized = normalize_term_unit(extra)
            if normalized not in filtered_parts:
                filtered_parts.append(normalized)
            if len(filtered_parts) >= 3:
                break

    filtered_parts = filtered_parts[:4]
    return "".join(filtered_parts + ["\uac12"])


def recommend_terms(description: str) -> tuple[list[str], list[str]]:
    """사용자 설명과 네이버 사전 후보 단어를 기반으로 추천 용어를 생성한다."""
    dictionary_words = fetch_naver_dictionary_words(description)
    prompt = f"""
사용자 설명과 네이버 사전 후보 단어를 바탕으로 표준 용어 후보 3개를 추천하세요.

[사용자 설명]
{description}

[네이버 사전 후보 단어]
{", ".join(dictionary_words) if dictionary_words else "없음"}

[제약 조건]
1. 반드시 3개만 추천합니다.
2. 각 용어는 단어 3~5개로 구성합니다.
3. 각 단어 사이는 공백 없이 붙여서 표기합니다.
4. 마지막 단어는 반드시 "값"입니다.
5. 각 구성 단어의 길이는 2~3글자입니다.
6. 되도록 한글 단어만 사용합니다.
7. 설명의 의미와 자연스럽게 맞아야 합니다.
8. 서로 다른 후보를 제안합니다.
9. JSON 배열 문자열만 반환합니다. 예: ["고객식별번호값", "고객고유정보값", "고객관리기준값"]
"""

    try:
        response = get_llm_client().chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-5"),
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 데이터 표준 용어를 설계하는 전문가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or "[]"
        parsed = json.loads(content)
        if not isinstance(parsed, list):
            raise ValueError("추천 결과 형식이 올바르지 않습니다.")

        recommendations = []
        for item in parsed:
            if not isinstance(item, str):
                continue
            normalized = sanitize_recommended_term(item)
            if normalized not in recommendations:
                recommendations.append(normalized)
        while len(recommendations) < 3:
            for fallback in fallback_recommend_terms(description, dictionary_words):
                normalized = sanitize_recommended_term(fallback)
                if normalized not in recommendations:
                    recommendations.append(normalized)
                if len(recommendations) >= 3:
                    break
        return recommendations[:3], dictionary_words
    except Exception:
        return fallback_recommend_terms(description, dictionary_words), dictionary_words


def main() -> None:
    st.set_page_config(page_title="표준용어 조회", page_icon="📘", layout="wide")
    st.title("표준용어 조회")
    st.write("설명 정보를 입력한 뒤 의미적으로 가장 가까운 표준용어를 조회할 수 있습니다.")

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []
    if "recommended_terms" not in st.session_state:
        st.session_state["recommended_terms"] = []
    if "dictionary_words" not in st.session_state:
        st.session_state["dictionary_words"] = []
    if "search_click_count" not in st.session_state:
        st.session_state["search_click_count"] = 0

    query = st.text_area(
        "설명 정보 입력",
        placeholder="예: 고객을 고유하게 식별하기 위한 번호",
        height=140,
    )

    search_col, recommend_col = st.columns([1, 1])
    search_clicked = search_col.button("표준용어 조회", type="primary", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.warning("조회할 설명 정보를 입력해주세요.")
        else:
            with st.spinner("FAISS에서 유사한 표준용어를 검색하는 중입니다..."):
                st.session_state["search_results"] = search_standard_terms(query.strip())
            st.session_state["search_click_count"] += 1

    recommend_enabled = st.session_state["search_click_count"] >= 3
    recommend_clicked = recommend_col.button(
        "용어 추천",
        use_container_width=True,
        disabled=not recommend_enabled,
    )

    if recommend_clicked:
        if not query.strip():
            st.warning("추천할 설명 정보를 입력해주세요.")
        else:
            with st.spinner("설명과 네이버 사전 단어를 바탕으로 용어를 추천하는 중입니다..."):
                recommended_terms, dictionary_words = recommend_terms(query.strip())
                st.session_state["recommended_terms"] = recommended_terms
                st.session_state["dictionary_words"] = dictionary_words

    if not recommend_enabled:
        remaining_count = 3 - st.session_state["search_click_count"]
        st.caption(
            f"용어 추천은 표준용어 조회를 3회 이상 실행한 뒤 사용할 수 있습니다. "
            f"남은 조회 횟수: {max(remaining_count, 0)}회"
        )
    else:
        st.caption(
            f"표준용어 조회 실행 횟수: {st.session_state['search_click_count']}회"
        )

    result_left_col, result_right_col = st.columns([1, 1], gap="large")

    with result_left_col:
        st.subheader("조회 결과")
        if st.session_state["search_results"]:
            for index, item in enumerate(st.session_state["search_results"], start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. 표준용어명:** {item['standard_term_name']}")
                    st.markdown(f"**설명:** {item['description']}")
                    st.caption(
                        f"유사도: {item['similarity']:.4f} | 거리(score): {item['distance']:.4f}"
                    )
        else:
            st.info("표준용어 조회 결과가 여기에 표시됩니다.")

    with result_right_col:
        st.subheader("추천 결과")
        if st.session_state["recommended_terms"]:
            if st.session_state["dictionary_words"]:
                st.caption(
                    "네이버 사전 후보 단어: "
                    + ", ".join(st.session_state["dictionary_words"])
                )
            for index, term in enumerate(st.session_state["recommended_terms"], start=1):
                with st.container(border=True):
                    st.markdown(f"**{index}. 추천용어:** {term}")
        else:
            st.info("용어 추천 결과가 여기에 표시됩니다.")


if __name__ == "__main__":
    main()
