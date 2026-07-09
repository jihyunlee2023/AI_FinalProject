"""LangGraph StateGraph 조립.

노드 책임·조건부 분기·반복 루프·설계 근거는 docs/WORKFLOW.md 를 참조.
"""

import logging
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.config import CHAT_MODEL, CHECKPOINT_DB_PATH, LOGGER_NAME, MAX_TOOL_ROUNDS
from app.learning import build_weakness_report, grade_answer, select_quiz
from app.middleware import InputGuardrail
from app.state import AgentState, AnswerCard
from app.tools import TOOLS
from app.vectorstore import get_exams_store

logger = logging.getLogger(LOGGER_NAME)

_agent_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0).bind_tools(TOOLS)


ROUTER_PROMPT = """당신은 한국사 학습 Agent '사관'의 라우터입니다.
사용자의 마지막 발화를 아래 네 의도 중 하나로 분류하세요.

- quiz    : 문제를 내달라는 요청 ("문제 내줘", "조선 후기 경제 문제 줘", "한 문제 더")
- grade   : 방금 낸 문제에 대한 답안 제출 (숫자 하나 "3", "3번", "정답 2번")
- weakness: 자신의 약점/취약점을 묻는 요청 ("나 뭐가 약해?", "약점 알려줘", "오답노트 보여줘")
- concept : 그 외 한국사 개념·사건·인물·연도를 묻는 질문 ("광복은 몇 년도야?", "탕평책이 뭐야?")

반드시 quiz / grade / weakness / concept 중 하나만 출력하세요."""

AGENT_PROMPT = """당신은 한국사 학습 Agent '사관'의 리서처입니다.
사용자의 개념 질문에 답하기 위해 필요한 근거를 도구로 수집하세요.

도구 사용 규칙:
1. 개념·사건·연도·인물 질문에는 먼저 search_history_source 로 개념·사료 DB를 검색하세요.
2. '~기출 보여줘', '~문제 예시'처럼 기출 사례를 원하면 search_past_exams 를 쓰세요.
3. 두 사건의 연도 차이('몇 년 차이')를 물으면 calculate_year_gap 을 쓰세요.
4. 개념 DB 결과가 NO_RESULT 이거나 지엽적 용어라 확인이 더 필요하면 search_wikipedia 로 보완하세요.
5. 충분한 근거가 모였거나 이미 대화 이력에 근거가 있으면 도구를 더 부르지 말고 종료하세요.
6. 단순 인사나 짧은 후속 반응에는 도구가 필요 없습니다."""

ANSWER_PROMPT = """당신은 친절한 한국사 선생님 '사관'입니다.

답변 규칙:
1. 반드시 아래 '수집된 근거 자료'에 있는 내용만으로 설명하세요. 근거에 없는 내용을 지어내지 마세요.
2. 존댓말로, 핵심을 짚어 명확하게 설명하세요.
3. related_topics 는 방금 답변과 이어지는, 학생이 더 궁금해할 한국사 질문 3개로 만드세요.
4. 이전 대화 맥락이 있으면 자연스럽게 이어받으세요."""

NOSOURCE_MESSAGE = (
    "음, 이건 제 개념 노트에도 없고 위키백과에서도 근거를 찾지 못했어요. "
    "확실하지 않은 걸 지어내서 알려드리긴 싫어서, 솔직히 '아직 모른다'고 말씀드릴게요. "
    "질문을 조금 더 구체적이거나 일반적인 표현으로 바꿔서 다시 물어봐 주세요!"
)


class Intent(BaseModel):
    """router 분류 결과의 구조화 출력 스키마."""

    intent: str = Field(description="quiz / grade / weakness / concept 중 하나")


# LCEL 체인 (ChatPromptTemplate | model) — 프롬프트·모델·OutputParser 를 Runnable 로 조합
_router_chain = ChatPromptTemplate.from_messages(
    [("system", ROUTER_PROMPT), ("human", "{question}")]
) | ChatOpenAI(model=CHAT_MODEL, temperature=0).with_structured_output(Intent)

_ANSWER_HUMAN = "{dialogue_context}[이번 질문]\n{question}\n\n[수집된 근거 자료]\n{evidence}"
_answer_chain = ChatPromptTemplate.from_messages(
    [("system", ANSWER_PROMPT), ("human", _ANSWER_HUMAN)]
) | ChatOpenAI(model=CHAT_MODEL, temperature=0.4).with_structured_output(AnswerCard)


# ── 노드 ──────────────────────────────────────────────────
def guard_node(state: AgentState) -> dict:
    """입력을 모델에 보내기 전 가드레일 검증."""
    ok, reason = InputGuardrail.validate(state["question"])
    logger.info("가드레일: ok=%s q=%r", ok, state["question"][:40])
    return {"blocked": not ok, "blocked_reason": reason}


def blocked_reply_node(state: AgentState) -> dict:
    """가드레일에 걸린 요청 안내."""
    return {
        "messages": [AIMessage(content=state["blocked_reason"])],
        "card": None,
        "card_type": "blocked",
    }


def router_node(state: AgentState) -> dict:
    """LLM 체인으로 사용자 의도를 4가지 중 하나로 분류."""
    result: Intent = _router_chain.invoke({"question": state["question"]})
    intent = result.intent.strip().lower()
    if intent not in {"quiz", "grade", "weakness", "concept"}:
        intent = "concept"
    logger.info("의도 분류: %s", intent)
    return {"intent": intent}


def quiz_node(state: AgentState) -> dict:
    """출제 — 요청/오답노트 기반으로 기출을 선택. last_quiz 에 저장해 다음 채점에 대비."""
    card = select_quiz(
        get_exams_store(),
        state["question"],
        state.get("wrong_tags", {}),
        state.get("last_quiz"),
    )
    return {
        "messages": [AIMessage(content=_format_quiz_message(card))],
        "card": card.model_dump(),
        "card_type": "quiz",
        "last_quiz": card.model_dump(),
    }


def grade_node(state: AgentState) -> dict:
    """채점 — last_quiz 와 대조. 오답이면 wrong_tags(오답노트)를 누적 갱신."""
    last_quiz = state.get("last_quiz")
    if not last_quiz:
        return {
            "messages": [AIMessage(content="아직 낸 문제가 없어요. '문제 내줘'라고 먼저 요청해 주세요!")],
            "card": None,
            "card_type": "grade",
        }

    card, weak_tag = grade_answer(last_quiz, state["question"])
    updates: dict = {
        "messages": [AIMessage(content=card.explanation)],
        "card": card.model_dump(),
        "card_type": "grade",
    }
    if weak_tag:
        wrong_tags = dict(state.get("wrong_tags", {}))
        wrong_tags[weak_tag] = wrong_tags.get(weak_tag, 0) + 1
        updates["wrong_tags"] = wrong_tags
        logger.info("오답노트 갱신: %s → %d", weak_tag, wrong_tags[weak_tag])
    return updates


def weakness_node(state: AgentState) -> dict:
    """약점 진단 — 오답노트 집계 + 빈출 통계 보강."""
    report = build_weakness_report(state.get("wrong_tags", {}))
    lines = ["📊 약점 진단 결과"]
    if report.weak_areas:
        lines.append("자주 틀린 영역: " + ", ".join(report.weak_areas))
    lines.append(report.suggestion)
    return {
        "messages": [AIMessage(content="\n".join(lines))],
        "card": report.model_dump(),
        "card_type": "weakness",
    }


def _messages_for_agent(state: AgentState) -> list:
    """agent 판단용 메시지 구성.

    과거 턴은 완결된 사람↔AI 메시지만, 현재 턴은 tool 메시지 포함 전체를 유지한다.
    (과거 턴의 tool_call↔ToolMessage 쌍 재전송으로 인한 OpenAI 400 에러 방지 — WORKFLOW.md 참조)
    """
    messages = state["messages"]
    last_human_idx = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    past = [
        m
        for m in messages[:last_human_idx]
        if isinstance(m, HumanMessage)
        or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
    ]
    current = messages[last_human_idx:]
    return [SystemMessage(content=AGENT_PROMPT), *past, *current]


def agent_node(state: AgentState) -> dict:
    """개념 질문 처리 — LLM 이 Tool 호출 여부·종류를 자율 결정(ReAct)."""
    response = _agent_llm.invoke(_messages_for_agent(state))
    n = len(getattr(response, "tool_calls", []) or [])
    logger.info("agent 판단: tool_calls=%d (round=%d)", n, state.get("iterations", 0))
    return {"messages": [response]}


_tool_node = ToolNode(TOOLS)


def tools_node(state: AgentState) -> dict:
    """Tool 실행 + 반복 횟수 증가(self-loop 제한)."""
    result = _tool_node.invoke(state)
    return {"messages": result["messages"], "iterations": state.get("iterations", 0) + 1}


def _collect_turn_evidence(state: AgentState) -> tuple[str, str, bool]:
    """이번 턴 ToolMessage 근거를 모아 (근거 텍스트, 출처, 유무)를 계산.

    출처는 LLM 자기보고가 아니라 실제 Tool 실행 기록으로 코드에서 확정한다.
    """
    parts: list[str] = []
    hits: set[str] = set()

    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            if "NO_RESULT" in content or not content.strip():
                continue
            parts.append(f"[{msg.name}]\n{content}")
            hits.add(msg.name)

    evidence = "\n\n====\n\n".join(reversed(parts))
    if "search_history_source" in hits:
        source = "history_notes"
    elif "search_past_exams" in hits:
        source = "past_exams"
    elif "search_wikipedia" in hits:
        source = "wikipedia"
    elif "calculate_year_gap" in hits:
        source = "calculation"
    else:
        source = "unknown"
    return evidence, source, bool(parts)


def _recent_dialogue(state: AgentState, max_turns: int = 3) -> str:
    """직전 대화 몇 턴을 텍스트로 요약해 멀티턴 맥락으로 전달."""
    pairs: list[str] = []
    q = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            q = str(msg.content)
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and q:
            pairs.append(f"Q: {q}\nA: {str(msg.content)[:120]}")
            q = None
    return "\n\n".join(pairs[-max_turns:])


def answer_node(state: AgentState) -> dict:
    """근거로 AnswerCard(구조화 출력)를 생성. 근거 있을 때만 진입."""
    evidence, source, _ = _collect_turn_evidence(state)
    dialogue = _recent_dialogue(state)
    context_block = f"[이전 대화 맥락]\n{dialogue}\n\n" if dialogue else ""

    try:
        card: AnswerCard = _answer_chain.invoke(
            {"dialogue_context": context_block, "question": state["question"], "evidence": evidence}
        )
    except Exception as exc:  # noqa: BLE001 — 구조화 실패 시 근거 원문 폴백
        logger.error("AnswerCard 생성 실패: %s", exc)
        card = AnswerCard(
            title="근거는 찾았는데 정리에 실패했어요",
            explanation=f"찾은 자료를 그대로 보여드릴게요.\n\n{evidence[:400]}",
            source=source,  # type: ignore[arg-type]
            is_verified=True,
            related_topics=["다시 물어보기", "더 짧게 질문하기", "다른 주제 물어보기"],
        )

    card.source = source  # type: ignore[assignment]
    card.is_verified = True
    logger.info("AnswerCard 생성: source=%s", source)
    return {
        "messages": [AIMessage(content=f"{card.title}\n\n{card.explanation}")],
        "card": card.model_dump(),
        "card_type": "answer",
        "context": evidence,
        "source": source,
    }


def nosource_node(state: AgentState) -> dict:
    """근거를 못 찾았을 때 정직하게 '모른다'고 답하는 노드(환각 방지)."""
    logger.info("근거 없음 — nosource: q=%r", state["question"][:40])
    card = AnswerCard(
        title="이건 아직 근거를 찾지 못했어요",
        explanation=NOSOURCE_MESSAGE,
        source="unknown",
        is_verified=False,
        related_topics=["광복은 몇 년도야?", "탕평책이 뭐야?", "임진왜란은 언제 일어났어?"],
    )
    return {
        "messages": [AIMessage(content=NOSOURCE_MESSAGE)],
        "card": card.model_dump(),
        "card_type": "answer",
        "source": "unknown",
    }


def _format_quiz_message(card) -> str:
    """QuizCard 를 채팅 텍스트(fallback 표시용)로 변환."""
    lines = [f"[{card.era} · {card.topic}]", "", card.question, "", *card.choices, ""]
    lines.append("정답 번호를 입력해 주세요! (근거: " + card.source + ")")
    return "\n".join(lines)


# ── 조건부 분기(conditional edge) 함수 ────────────────────
def route_after_guard(state: AgentState) -> str:
    """분기 ① 가드레일 차단 여부."""
    return "blocked_reply" if state["blocked"] else "router"


def route_after_router(state: AgentState) -> str:
    """분기 ② 의도 기반 4-way 라우팅."""
    intent = state["intent"]
    return {"quiz": "quiz", "grade": "grade", "weakness": "weakness"}.get(intent, "agent")


def route_after_agent(state: AgentState) -> str:
    """분기 ③ Tool 반복 여부 + 근거 유무."""
    last = state["messages"][-1]
    wants_tools = bool(getattr(last, "tool_calls", None))
    over_limit = state.get("iterations", 0) >= MAX_TOOL_ROUNDS

    if wants_tools and not over_limit:
        return "tools"
    if wants_tools and over_limit:
        logger.warning("Tool 반복 한도(%d) 도달 — 답변 단계로", MAX_TOOL_ROUNDS)

    _, _, has_evidence = _collect_turn_evidence(state)
    return "answer" if has_evidence else "nosource"


# ── 그래프 조립 ───────────────────────────────────────────
def build_graph():
    """StateGraph 를 조립하고 checkpointer(멀티턴 메모리)와 함께 컴파일."""
    builder = StateGraph(AgentState)

    builder.add_node("guard", guard_node)
    builder.add_node("blocked_reply", blocked_reply_node)
    builder.add_node("router", router_node)
    builder.add_node("quiz", quiz_node)
    builder.add_node("grade", grade_node)
    builder.add_node("weakness", weakness_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("answer", answer_node)
    builder.add_node("nosource", nosource_node)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard", route_after_guard, {"blocked_reply": "blocked_reply", "router": "router"}
    )
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {"quiz": "quiz", "grade": "grade", "weakness": "weakness", "agent": "agent"},
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "answer": "answer", "nosource": "nosource"},
    )
    builder.add_edge("tools", "agent")  # 반복 루프

    for terminal in ("blocked_reply", "quiz", "grade", "weakness", "answer", "nosource"):
        builder.add_edge(terminal, END)

    # SqliteSaver: 대화 이력·오답노트를 파일 DB에 영속화 → 서버를 껐다 켜도 유지.
    # FastAPI 는 멀티스레드로 요청을 처리하므로 check_same_thread=False.
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
