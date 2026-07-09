"""Middleware 계층 — 운영 안정성 확보.

InputGuardrail(입력 검증, before_model 역할) + tool_middleware(로깅·재시도, wrap_tool_call 역할).
설계 근거는 docs/WORKFLOW.md 참조.
"""

import functools
import logging
import re
import time

from app.config import LOGGER_NAME, MAX_INPUT_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(LOGGER_NAME)


# ── 1. 입력 검증 가드레일 ─────────────────────────────────
# 학습 서비스 목적에 맞지 않는 위험/부적절 요청 패턴
_BLOCKED_PATTERNS: list[tuple[str, str]] = [
    (r"(주민등록번호|여권번호|계좌번호|카드번호)", "개인정보와 관련된 질문에는 답변할 수 없어요."),
    (r"(폭탄|마약|해킹|무기)\s*(제조|만드는|만들|구하는)", "위험하거나 불법적인 정보는 다룰 수 없어요."),
    (r"(자살|자해)\s*(방법|하는\s*법)", "힘든 일이 있다면 자살예방상담전화 1393에 연락해 보세요."),
]


class InputGuardrail:
    """사용자 입력을 모델에 전달하기 전에 검증하는 가드레일.

    graph 의 진입 노드(guard)에서 호출된다.
    """

    @staticmethod
    def validate(text: str) -> tuple[bool, str]:
        """유효하면 (True, ""), 차단해야 하면 (False, 사유)를 반환."""
        stripped = (text or "").strip()

        if not stripped:
            return False, "질문이 비어 있어요. 궁금한 걸 입력해 주세요!"

        if len(stripped) > MAX_INPUT_LENGTH:
            return False, f"질문이 너무 길어요. {MAX_INPUT_LENGTH}자 이내로 줄여 주세요."

        for pattern, reason in _BLOCKED_PATTERNS:
            if re.search(pattern, stripped):
                logger.warning("가드레일 차단: pattern=%s input=%r", pattern, stripped[:50])
                return False, reason

        return True, ""


# ── 2. Tool 호출 로깅 + 재시도 미들웨어 ───────────────────
def tool_middleware(max_retries: int = 1, fallback: str = "NO_RESULT"):
    """Tool 을 감싸 로깅·예외처리·백오프 재시도를 수행하는 데코레이터.

    최종 실패 시 그래프가 죽지 않도록 fallback 문자열을 반환한다.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__name__
            for attempt in range(1, max_retries + 2):  # 최초 1회 + 재시도
                start = time.perf_counter()
                try:
                    logger.info("[Tool 호출] %s (attempt=%d) args=%s", name, attempt, args or kwargs)
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    logger.info("[Tool 완료] %s (%.2fs) → %.60r", name, elapsed, str(result))
                    return result
                except Exception as exc:  # noqa: BLE001 — 운영 안정성 위해 광범위 캐치
                    elapsed = time.perf_counter() - start
                    logger.error(
                        "[Tool 실패] %s (%.2fs, attempt=%d) error=%s", name, elapsed, attempt, exc
                    )
                    if attempt >= max_retries + 1:
                        return fallback
                    time.sleep(0.4 * attempt)  # 선형 백오프 후 재시도
            return fallback  # 논리상 도달하지 않는 안전망

        return wrapper

    return decorator
