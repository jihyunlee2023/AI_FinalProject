"""RAG 파이프라인 — 기출·개념 두 FAISS 벡터스토어 구축/로드.

전처리 → 임베딩 → 인덱싱을 두 원천(past_exams.jsonl, history_notes.txt)에 각각 수행하고
결과를 로컬(data/faiss_*)에 캐시한다. 전처리 방식·설계 근거는 docs/WORKFLOW.md 참조.
"""

import json
import logging

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import (
    EMBEDDING_MODEL,
    EXAMS_INDEX_DIR,
    HISTORY_NOTES_PATH,
    LOGGER_NAME,
    NOTES_INDEX_DIR,
    PAST_EXAMS_PATH,
)

logger = logging.getLogger(LOGGER_NAME)


def _embeddings() -> OpenAIEmbeddings:
    """임베딩 모델 인스턴스 (모듈 공통)."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


# ── 데이터 전처리: 원천 파일 → Document 리스트 ────────────
def _load_exam_documents() -> list[Document]:
    """past_exams.jsonl 파싱 → Document. 지문+선택지+해설을 임베딩, 나머지는 metadata 로 분리."""
    documents: list[Document] = []
    raw = PAST_EXAMS_PATH.read_text(encoding="utf-8")

    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("기출 파싱 실패(%d행) — 건너뜀", line_no)
            continue

        content = "\n".join(
            [
                f"[{item['era']}·{item['topic']}] {item['question']}",
                *item["choices"],
                f"해설: {item['explanation']}",
            ]
        )
        documents.append(Document(page_content=content, metadata=item))

    logger.info("기출 문제 %d건 전처리 완료", len(documents))
    return documents


def _load_note_documents() -> list[Document]:
    """history_notes.txt 를 파싱해 Document 로 변환. ('제목 ::: 본문')"""
    documents: list[Document] = []
    raw = HISTORY_NOTES_PATH.read_text(encoding="utf-8")

    for line in raw.splitlines():
        line = line.strip()
        if not line or ":::" not in line:
            continue  # 빈 줄·형식 불일치 줄은 스킵
        title, body = (part.strip() for part in line.split(":::", maxsplit=1))
        documents.append(
            Document(page_content=f"{title}\n{body}", metadata={"title": title})
        )

    logger.info("개념·사료 %d건 전처리 완료", len(documents))
    return documents


# ── 인덱스 구축/로드 (캐시 우선) ──────────────────────────
def _get_or_build(index_dir, loader) -> FAISS:
    """캐시가 있으면 로드, 없으면 새로 구축 후 저장하는 공통 헬퍼."""
    embeddings = _embeddings()
    if index_dir.exists():
        logger.info("FAISS 캐시 로드: %s", index_dir.name)
        return FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )

    logger.info("FAISS 신규 구축 시작: %s", index_dir.name)
    store = FAISS.from_documents(loader(), embeddings)
    store.save_local(str(index_dir))
    logger.info("FAISS 저장 완료: %s", index_dir.name)
    return store


# 서버 기동 시 1회 구축 후 재사용하기 위한 모듈 캐시
_exams_store: FAISS | None = None
_notes_store: FAISS | None = None


def get_exams_store() -> FAISS:
    """기출 벡터스토어(싱글턴)."""
    global _exams_store
    if _exams_store is None:
        _exams_store = _get_or_build(EXAMS_INDEX_DIR, _load_exam_documents)
    return _exams_store


def get_notes_store() -> FAISS:
    """개념·사료 벡터스토어(싱글턴)."""
    global _notes_store
    if _notes_store is None:
        _notes_store = _get_or_build(NOTES_INDEX_DIR, _load_note_documents)
    return _notes_store


def init_vectorstores() -> None:
    """서버 시작 시 두 인덱스를 미리 로드해 첫 요청 지연을 방지."""
    get_exams_store()
    get_notes_store()
