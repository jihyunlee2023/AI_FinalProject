"""FastAPI 웹 서버 — 사관 Agent 그래프를 HTTP API 로 노출.

실행: uvicorn app.server:app --reload
엔드포인트: GET / (채팅 UI) · POST /api/chat · GET /api/diagram (Mermaid)
"""

import logging
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

# .env 의 OPENAI_API_KEY 로드 — 반드시 앱 import 보다 먼저
load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402

from app.config import LOGGER_NAME, STATIC_DIR  # noqa: E402
from app.graph import graph  # noqa: E402
from app.vectorstore import init_vectorstores  # noqa: E402

logger = logging.getLogger(LOGGER_NAME)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """서버 시작 시 두 FAISS 인덱스를 미리 구축/로드 (첫 요청 지연 방지)."""
    init_vectorstores()
    logger.info("사관 서버 준비 완료")
    yield


app = FastAPI(title="사관(史官) — 한국사 학습 Agent", lifespan=lifespan)


# ── 요청/응답 스키마 (Pydantic — FastAPI 본문 검증) ────────
class ChatRequest(BaseModel):
    message: str = Field(description="사용자 질문/답변")
    thread_id: str | None = Field(default=None, description="대화 세션 ID(없으면 발급)")
    # 답안 제출 시 '풀고 있는 문제'를 함께 보내면, 서버 재시작으로 인메모리
    # last_quiz 가 사라져도 채점이 성립한다(WORKFLOW.md 참조).
    quiz: dict | None = Field(default=None, description="채점 대상 QuizCard(선택)")


class ChatResponse(BaseModel):
    thread_id: str
    reply: str = ""                 # 채팅 말풍선에 표시할 텍스트
    card_type: str = ""             # quiz / grade / weakness / answer / blocked
    card: dict | None = None        # 구조화 카드(프론트 렌더링용)
    wrong_tags: dict = {}           # 현재 오답노트(사이드바 표시용)


def _new_turn_state(message: str) -> dict:
    """한 턴마다 초기화되는 입력 State. 메모리 필드(messages/wrong_tags/last_quiz)는
    checkpointer 가 이전 값을 유지하므로 여기서 덮어쓰지 않는다."""
    return {
        "messages": [HumanMessage(content=message)],
        "question": message,
        "intent": "",
        "context": "",
        "source": "",
        "iterations": 0,
        "blocked": False,
        "blocked_reason": "",
        "card": None,
        "card_type": "",
    }


@app.get("/")
async def index() -> FileResponse:
    """채팅 UI 페이지."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """질문을 Agent 그래프에 전달하고 구조화된 응답을 반환."""
    # 모든 사용자가 같은 스레드를 사용하여 오답노트 누적 저장
    # (회차 진행 중인 채팅만 session_id로 분리)
    session_id = req.thread_id or str(uuid.uuid4())
    memory_thread_id = "default_user"  # 오답노트 누적 저장용 고정 스레드
    config = {"configurable": {"thread_id": memory_thread_id}}

    input_state = _new_turn_state(req.message)
    if req.quiz:  # 클라이언트가 보낸 '풀고 있는 문제'를 채점 대상으로 주입
        input_state["last_quiz"] = req.quiz

    try:
        result = graph.invoke(input_state, config=config)
    except Exception as exc:  # noqa: BLE001 — API 키/네트워크 장애를 친화적으로 처리
        logger.exception("그래프 실행 실패: %s", exc)
        return ChatResponse(
            thread_id=session_id,
            reply="답변을 만드는 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요. (API 키·네트워크 확인)",
            card_type="error",
        )

    # 마지막 AI 메시지를 말풍선 텍스트로 사용
    reply = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, HumanMessage):
            break
        if getattr(msg, "content", "") and not getattr(msg, "tool_calls", None):
            reply = msg.content
            break

    return ChatResponse(
        thread_id=session_id,  # 클라이언트에 현재 세션 ID 반환
        reply=reply,
        card_type=result.get("card_type", ""),
        card=result.get("card"),
        wrong_tags=result.get("wrong_tags", {}),
    )


@app.get("/api/diagram", response_class=PlainTextResponse)
async def diagram() -> str:
    """LangGraph 가 자동 생성한 Mermaid 다이어그램 (문서·검증용)."""
    return graph.get_graph().draw_mermaid()
