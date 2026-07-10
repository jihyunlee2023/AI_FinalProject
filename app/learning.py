"""학습 도메인 로직 — 출제·채점·약점 진단의 결정론적 순수 로직.

LLM 자율 판단이 필요 없는 이 로직을 그래프에서 분리해, graph.py 는 흐름 조립에만
집중하게 한다(관심사 분리). 설계 근거는 docs/WORKFLOW.md 참조.
"""

import json
import logging
import random
import re
from collections import defaultdict

from app.config import EXAMS_TOP_K, LOGGER_NAME, PAST_EXAMS_PATH, TOPICS, TOPIC_STATS_PATH
from app.state import GradeCard, QuizCard, WeaknessReport

logger = logging.getLogger(LOGGER_NAME)


# ── 데이터 로드 (모듈 최초 import 시 1회) ──────────────────
def _load_all_exams() -> list[dict]:
    """past_exams.jsonl 전체를 리스트로 로드 (랜덤 출제·태그 필터용)."""
    items: list[dict] = []
    for line in PAST_EXAMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    logger.info("출제 풀 %d문항 로드", len(items))
    return items


def _load_freq_table() -> dict[tuple[str, str], int]:
    """topic_stats.json 을 (시대, 주제) → 최근 22회 출제 횟수 합계로 집계."""
    table: dict[tuple[str, str], int] = defaultdict(int)
    try:
        stats = json.loads(TOPIC_STATS_PATH.read_text(encoding="utf-8"))
        for row in stats:
            table[(row["era"], row["topic"])] += row.get("count", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("빈출 통계(topic_stats.json) 로드 실패 — 통계 보강 생략")
    return table


ALL_EXAMS: list[dict] = _load_all_exams()
FREQ_TABLE: dict[tuple[str, str], int] = _load_freq_table()


# ── 출제 (quiz) ───────────────────────────────────────────
# 사용자가 자연어로 쓰는 시대 표현 → 표준 era 값 (띄어쓰기·약칭 대응)
_ERA_ALIASES: dict[str, str] = {
    "선사": "선사", "구석기": "선사", "신석기": "선사", "청동기": "선사",
    "고조선": "고조선", "부여": "고조선", "삼한": "고조선",
    "삼국": "삼국", "고구려": "삼국", "백제": "삼국", "신라": "삼국", "가야": "삼국",
    "남북국": "남북국", "통일신라": "남북국", "통일 신라": "남북국", "발해": "남북국",
    "고려": "고려",
    "조선전기": "조선전기", "조선 전기": "조선전기",
    "조선후기": "조선후기", "조선 후기": "조선후기",
    "개항기": "개항기", "개항": "개항기", "구한말": "개항기", "대한제국": "개항기",
    "일제강점기": "일제강점기", "일제 강점기": "일제강점기", "일제": "일제강점기",
    "현대": "현대", "광복 이후": "현대",
}


def _detect_requested_tag(question: str) -> tuple[str | None, str | None]:
    """질문에서 명시적으로 요청한 (시대, 주제)를 추출. 없으면 (None, None)."""
    era = next((v for k, v in _ERA_ALIASES.items() if k in question), None)
    topic = next((t for t in TOPICS if t in question), None)
    return era, topic


# '취약 영역/약점 위주로 내달라'는 의도를 나타내는 표현
_WEAK_FOCUS_KEYWORDS = ("취약", "약점", "약한", "틀린", "오답", "복습")


def _wants_weak_focus(question: str) -> bool:
    """사용자가 취약 영역 집중 출제를 요청했는지 판별."""
    return any(kw in question for kw in _WEAK_FOCUS_KEYWORDS)


def _weakest_tag(wrong_tags: dict) -> tuple[str, str] | None:
    """오답노트에서 가장 많이 틀린 '시대·주제'를 (era, topic)으로 반환."""
    if not wrong_tags:
        return None
    tag = max(wrong_tags, key=wrong_tags.get)
    if "·" in tag:
        era, topic = tag.split("·", maxsplit=1)
        return era, topic
    return None


def select_quiz(
    exams_store, question: str, wrong_tags: dict, last_quiz: dict | None
) -> QuizCard:
    """기출 벡터DB에서 문항 1개를 선택해 출제.

    우선순위:
      ① 명시적으로 요청한 시대·주제 ("조선 후기 경제 문제 줘")
      ② 취약 영역 집중 요청 + 오답노트 존재 ("취약 영역 문제 줘", "약점 부분 내줘")
      ③ 그 외 일반 요청 ("문제 내줘") → 랜덤
    실제 기출 DB에서 '검색해 선택'하므로 환각이 없다. 직전 문항은 제외.
    """
    last_q = (last_quiz or {}).get("question")

    req_era, req_topic = _detect_requested_tag(question)
    if req_era or req_topic:
        target, reason = (req_era, req_topic), "요청"
    elif _wants_weak_focus(question) and (weak := _weakest_tag(wrong_tags)) is not None:
        target, reason = weak, "약점"
    else:
        target, reason = None, "랜덤"

    if target:
        era, topic = target
        query = f"{era or ''} {topic or ''} 관련 기출 문제".strip()
        candidates = [
            doc.metadata for doc in exams_store.similarity_search(query, k=EXAMS_TOP_K + 5)
        ]
        matched = [
            m
            for m in candidates
            if (era is None or m.get("era") == era)
            and (topic is None or m.get("topic") == topic)
        ] or candidates
        pool = [m for m in matched if m.get("question") != last_q] or matched
        logger.info("%s 기반 출제: 대상=%s·%s, 후보=%d", reason, era, topic, len(pool))
    else:
        pool = [m for m in ALL_EXAMS if m.get("question") != last_q]
        logger.info("랜덤 출제: 풀=%d", len(pool))

    item = random.choice(pool)
    stats = item.get("exam_stats", {})
    freq_note = f" (최근 22회 중 {stats['count']}회 출제 유형)" if stats.get("count") else ""

    return QuizCard(
        question=item["question"],
        choices=item["choices"],
        answer_index=item["answer"],
        explanation=item["explanation"],
        era=item["era"],
        topic=item["topic"],
        source=item.get("source", "한능검 기출 유형 · 자체 제작") + freq_note,
    )


# ── 채점 (grade) ──────────────────────────────────────────
def _parse_submitted_index(text: str) -> int | None:
    """사용자 답에서 선택 번호(1~5)를 추출. ('3', '3번', '④' 등)"""
    circled = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5}
    for mark, num in circled.items():
        if mark in text:
            return num
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else None


def grade_answer(last_quiz: dict, user_text: str) -> tuple[GradeCard, str | None]:
    """last_quiz 정답과 사용자 답을 대조. 반환: (GradeCard, 오답이면 취약 태그 / 정답이면 None)."""
    submitted = _parse_submitted_index(user_text)
    correct = last_quiz["answer_index"]
    weak_tag = f"{last_quiz['era']}·{last_quiz['topic']}"

    if submitted is None:
        # 번호를 못 읽으면 오답으로 집계하지 않고 재입력을 유도한다.
        card = GradeCard(
            is_correct=False,
            submitted_index=0,
            correct_index=correct,
            explanation="답을 숫자(1~5)로 인식하지 못했어요. 선택지 번호로 다시 답해 주세요.",
            source=last_quiz.get("source", ""),
            weak_tag=weak_tag,
        )
        return card, None

    is_correct = submitted == correct
    verdict = "정답입니다! 🎉" if is_correct else f"오답입니다. 정답은 {correct}번이에요."
    card = GradeCard(
        is_correct=is_correct,
        submitted_index=submitted,
        correct_index=correct,
        explanation=f"{verdict}\n\n{last_quiz['explanation']}",
        source=last_quiz.get("source", ""),
        weak_tag=weak_tag,
    )
    return card, (None if is_correct else weak_tag)


# ── 약점 진단 (weakness) ──────────────────────────────────
def build_weakness_report(wrong_tags: dict) -> WeaknessReport:
    """오답노트를 집계해 약점 리포트를 만든다. 빈출 통계로 학습 제안을 보강."""
    total = sum(wrong_tags.values())
    if total == 0:
        return WeaknessReport(
            weak_areas=[],
            total_wrong=0,
            suggestion="아직 틀린 문제가 없어요. 먼저 문제를 몇 개 풀어 보면 약점을 진단해 드릴게요!",
        )

    ranked = sorted(wrong_tags.items(), key=lambda kv: kv[1], reverse=True)[:3]
    weak_areas = [f"{tag} ({cnt}회 오답)" for tag, cnt in ranked]

    top_tag = ranked[0][0]
    era, _, topic = top_tag.partition("·")
    freq = FREQ_TABLE.get((era, topic), 0)
    freq_hint = (
        f" 특히 '{top_tag}'는 최근 22회 기출에서 약 {freq}회 출제된 빈출 영역이라 우선 보완이 필요해요."
        if freq
        else f" 특히 '{top_tag}' 영역을 집중적으로 복습해 보세요."
    )

    return WeaknessReport(
        weak_areas=weak_areas,
        total_wrong=total,
        suggestion=f"지금까지 {total}문제를 틀렸어요.{freq_hint} '취약 영역 문제 줘'(또는 '{top_tag} 문제 줘')라고 하시면 그 영역을 집중 출제해 드릴게요.",
    )
