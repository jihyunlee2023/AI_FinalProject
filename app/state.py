"""LangGraph 공유 상태(State) + 구조화 출력 스키마(Pydantic).

State 필드 설계와 메모리 유지 방식은 docs/WORKFLOW.md 의 '상태 설계' 절을 참조.
"""

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ── 구조화 출력 카드 (with_structured_output / model_dump 로 사용) ──
class QuizCard(BaseModel):
    """출제 결과 — quiz_node 가 생성."""

    question: str = Field(description="문항 지문")
    choices: list[str] = Field(description="4지선다 선택지", min_length=4, max_length=5)
    answer_index: int = Field(description="정답 번호(1부터 시작)", ge=1, le=5)
    explanation: str = Field(description="정답 해설")
    era: str = Field(description="시대 분류(예: 조선후기)")
    topic: str = Field(description="주제 분류(예: 경제)")
    source: str = Field(description="근거 출처(기출 회차/유형). 없으면 '확인된 자료 없음'")


class GradeCard(BaseModel):
    """채점 결과 — grade_node 가 생성."""

    is_correct: bool = Field(description="정답 여부")
    submitted_index: int = Field(description="사용자가 제출한 번호")
    correct_index: int = Field(description="실제 정답 번호")
    explanation: str = Field(description="해설")
    source: str = Field(description="근거 출처")
    weak_tag: str = Field(description="'조선후기·경제' 형태의 취약 태그(오답 시 오답노트에 누적)")


class WeaknessReport(BaseModel):
    """약점 진단 — weakness_node 가 생성."""

    weak_areas: list[str] = Field(description="자주 틀린 시대·주제 상위 N개")
    total_wrong: int = Field(description="누적 오답 수")
    suggestion: str = Field(description="다음 학습 제안")


class AnswerCard(BaseModel):
    """개념 질문 답변 — answer_node 가 생성."""

    title: str = Field(description="핵심을 한 줄로 요약한 제목")
    explanation: str = Field(description="근거에 기반한 친절한 설명(존댓말)")
    source: Literal[
        "past_exams", "history_notes", "wikipedia", "calculation", "unknown"
    ] = Field(description="근거 출처. 실제 Tool 실행 기록으로 코드에서 확정한다")
    is_verified: bool = Field(description="검색 근거에 기반한 답변이면 True")
    related_topics: list[str] = Field(
        description="이어서 궁금해할 만한 관련 질문 3개",
        min_length=3,
        max_length=3,
    )


# ── StateGraph 전체에서 공유되는 상태 ─────────────────────
class AgentState(TypedDict):
    """모든 노드가 읽고 쓰는 공유 상태."""

    # 대화 이력 — add_messages 리듀서가 새 메시지를 기존 리스트에 병합(append)
    messages: Annotated[list, add_messages]

    # ── 이번 턴 입력/분류 ────────────────────────────────
    question: str            # 이번 턴 사용자 질문
    intent: str              # router 분류 결과: quiz / grade / weakness / concept

    # ── 메모리(thread별 누적, checkpointer로 유지) ───────
    wrong_tags: dict         # {"조선후기·경제": 3, ...} 오답 누적 (오답노트)
    last_quiz: Optional[dict]  # 직전 출제 문제(QuizCard.model_dump()) — 채점 대조용

    # ── 개념질문(agent) 작업용 ───────────────────────────
    context: str             # 검색으로 수집한 근거 텍스트
    source: str              # 근거 출처
    iterations: int          # agent ⇄ tools 반복 횟수(한도 제한)

    # ── 가드레일 ─────────────────────────────────────────
    blocked: bool            # 입력이 차단되었는지
    blocked_reason: str      # 차단 사유(사용자 안내용)

    # ── 최종 출력 ────────────────────────────────────────
    card: Optional[dict]     # 최종 구조화 결과(QuizCard/GradeCard/... .model_dump())
    card_type: str           # 카드 종류: quiz / grade / weakness / answer / blocked
