"""Agent 가 자율적으로 선택·실행하는 Tool 4종 (벡터검색 2 + 외부검색 1 + 계산 1).

각 함수의 docstring 이 곧 LLM 의 Tool 선택 근거이므로 '언제 쓸지'를 명확히 기술한다.
검색 Tool 은 tool_middleware 로 감싸 로깅·예외처리·재시도가 자동 적용된다.
Tool 구성·선택 흐름은 docs/WORKFLOW.md 참조.
"""

import logging

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool

from app.config import (
    EXAMS_TOP_K,
    LOGGER_NAME,
    NOTES_SCORE_THRESHOLD,
    NOTES_TOP_K,
)
from app.middleware import tool_middleware
from app.vectorstore import get_exams_store, get_notes_store

logger = logging.getLogger(LOGGER_NAME)

_wiki_runner = WikipediaQueryRun(  # 한국어, 상위 2건, 본문 1,500자 제한
    api_wrapper=WikipediaAPIWrapper(lang="ko", top_k_results=2, doc_content_chars_max=1500)
)


@tool
@tool_middleware(max_retries=1, fallback="NO_RESULT")
def search_past_exams(query: str) -> str:
    """한능검 기출 문제 벡터DB에서 질의와 유사한 문항을 검색한다.

    특정 시대·주제의 기출 문제 예시가 필요하거나("조선 후기 경제 기출 보여줘"),
    개념 설명에 실제 출제 사례를 곁들이고 싶을 때 사용한다.
    결과가 없으면 'NO_RESULT'를 반환한다.
    """
    results = get_exams_store().similarity_search(query, k=EXAMS_TOP_K)
    if not results:
        return "NO_RESULT"
    return "\n\n---\n\n".join(doc.page_content for doc in results)


@tool
@tool_middleware(max_retries=1, fallback="NO_RESULT")
def search_history_source(query: str) -> str:
    """한국사 개념·사료 벡터DB에서 질의와 관련된 설명을 검색한다.

    '광복은 몇 년도야?', '탕평책이 뭐야?'처럼 개념·사건·연도·인물을 묻는
    질문에 가장 먼저 사용한다. 거리 임계값으로 관련성을 필터링하며,
    관련 내용이 없으면 'NO_RESULT'를 반환한다.
    """
    scored = get_notes_store().similarity_search_with_score(query, k=NOTES_TOP_K)
    relevant = [doc for doc, score in scored if score <= NOTES_SCORE_THRESHOLD]
    if not relevant:
        best = scored[0][1] if scored else float("inf")
        logger.info("개념DB 관련 결과 없음 (best score=%.3f)", best)
        return "NO_RESULT"
    return "\n\n---\n\n".join(doc.page_content for doc in relevant)


@tool
@tool_middleware(max_retries=1, fallback="NO_RESULT")
def search_wikipedia(query: str) -> str:
    """한국어 위키백과에서 사실 정보를 검색한다.

    개념·사료 DB에 관련 내용이 없거나(NO_RESULT), 우리 노트에 없는
    지엽적 인물·지명·용어를 확인해야 할 때 보완 검색으로 사용한다.
    결과가 없으면 'NO_RESULT'를 반환한다.
    """
    result = _wiki_runner.run(query)
    if not result or not result.strip():
        return "NO_RESULT"
    return result


@tool
def calculate_year_gap(event1_year: int, event2_year: int) -> str:
    """두 역사적 사건(연도) 사이의 시간 차이를 계산한다.

    '임진왜란과 병자호란은 몇 년 차이야?'처럼 두 사건의 연도 간격을 묻는
    질문에 사용한다. 각 사건의 발생 연도를 정수로 넣으면 차이를 계산한다.
    (기원전은 음수로 입력)
    """
    gap = abs(event2_year - event1_year)
    return f"두 연도({event1_year}, {event2_year})의 차이는 {gap}년입니다."


TOOLS = [search_past_exams, search_history_source, search_wikipedia, calculate_year_gap]
