"""사관(史官) — 오답노트 기반 한국사능력검정 학습 Agent.

패키지 구성:
    config      : 공통 상수 · 모델명 · 경로 (설정 일원화)
    state       : AgentState(TypedDict) + 구조화 출력 카드 4종(Pydantic)
    middleware  : 입력 가드레일 + Tool 로깅/재시도 데코레이터
    vectorstore : 기출·개념 FAISS 인덱스 구축/로드 (RAG 원천)
    tools       : Agent가 자율 선택하는 Tool 4종
    graph       : LangGraph StateGraph 전체 흐름
    server      : FastAPI 웹 서버
"""
