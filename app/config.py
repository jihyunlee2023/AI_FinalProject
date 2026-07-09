"""프로젝트 전역 설정 — 모델명·경로·임계값을 한 곳에서 관리.

여러 모듈에 흩어지기 쉬운 상수를 이 파일로 모아 재사용성과 유지보수성을 높인다.
값을 바꾸고 싶으면 이 파일만 수정하면 된다. (예: 모델 교체, 반복 한도 조정)
"""

from pathlib import Path

# ── 경로 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

PAST_EXAMS_PATH = DATA_DIR / "past_exams.jsonl"        # 기출 문제(RAG 원천)
HISTORY_NOTES_PATH = DATA_DIR / "history_notes.txt"    # 개념·사료 요약(RAG 원천)
TOPIC_STATS_PATH = DATA_DIR / "topic_stats.json"       # 최근 22회차 빈출 통계
EXAMS_INDEX_DIR = DATA_DIR / "faiss_exams"             # 기출 FAISS 캐시
NOTES_INDEX_DIR = DATA_DIR / "faiss_notes"             # 개념 FAISS 캐시
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"   # 대화·오답노트 영속 저장(SqliteSaver)

# ── 모델 ──────────────────────────────────────────────────
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ── RAG 검색 파라미터 ─────────────────────────────────────
EXAMS_TOP_K = 3          # 기출 유사 문항 검색 개수
NOTES_TOP_K = 3          # 개념·사료 검색 개수
# FAISS 기본 L2 거리 — 값이 작을수록 유사. 이 값 이하만 '관련 있음'으로 인정.
# 개념 노트를 위키보다 우선 활용하도록 다소 관대하게 설정(짧은 키워드 질의 대응).
NOTES_SCORE_THRESHOLD = 1.35

# ── Agent 반복 루프 제어 ──────────────────────────────────
MAX_TOOL_ROUNDS = 3      # agent ⇄ tools 반복 상한 (무한루프 방지)

# ── 입력 가드레일 ─────────────────────────────────────────
MAX_INPUT_LENGTH = 300   # 질문 최대 길이(자)

# ── 도메인 상수: 한능검 시대·주제 분류 체계 ───────────────
ERAS = [
    "선사", "고조선", "삼국", "남북국", "고려",
    "조선전기", "조선후기", "개항기", "일제강점기", "현대",
]
TOPICS = ["정치", "경제", "사회", "문화", "대외관계"]

# 로거 이름 (모듈 전체 공유)
LOGGER_NAME = "sagwan"
