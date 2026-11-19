from datetime import datetime, timezone
import logging
from sqlalchemy import select
from db.connection import SessionLocal
from db.models import TwinProfile, TwinMemory, ChatMessage
from services.core.prompt_builder import build_system_prompt
from services.core.decay import compute_state
from services.core.crisis_resources import CRISIS_RESOURCES
from services.core.grief_analyzer import analyze_message
from services.core.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)
EMBED_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:8b"
CRISIS_ESCALATION_THRESHOLD = 3

def retrieve(query: str, twin_id: int, k: int = 3) -> str:
    try:
        q = get_ollama_client().embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    except Exception as error:
        logger.error(
            "Query embedding failed; continuing without retrieved memories (error_type=%s)",
            type(error).__name__,
        )
        return ""
    with SessionLocal() as s:
        rows = s.execute(
            select(TwinMemory.content)
            .where(TwinMemory.twin_id == twin_id, TwinMemory.embedding.is_not(None))
            .order_by(TwinMemory.embedding.cosine_distance(q))
            .limit(k)
        ).all()
    return "\n".join(r[0] for r in rows)

def chat(user_msg: str, twin_id: int, user_id: str) -> str:
    # 1. Вычислить актуальное состояние и обновить last_interaction_at
    current_state = compute_state(twin_id)
    with SessionLocal() as s:
        twin = s.get(TwinProfile, twin_id)
        if twin:
            if twin.current_state != current_state:
                twin.current_state = current_state
            twin.last_interaction_at = datetime.now(timezone.utc)
            s.commit()

    # 2. Если LOCKED — не вызываем модель
    if current_state == "LOCKED":
        return ""

    try:
        analysis = analyze_message(user_msg)
    except Exception as error:
        logger.error(
            "Message safety analysis failed; continuing without classification (error_type=%s)",
            type(error).__name__,
        )
        analysis = None

    with SessionLocal() as s:
        previous_score = s.execute(
            select(ChatMessage.crisis_score)
            .where(ChatMessage.user_id == user_id, ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(1)
        ).scalar_one_or_none()

    previous_score = previous_score or 0
    if analysis is None:
        crisis_score = previous_score
        crisis_signal = None
        grief_orientation = None
    elif analysis.crisis_signal:
        crisis_score = previous_score + 1
        crisis_signal = True
        grief_orientation = None
    else:
        crisis_score = max(0, previous_score - 1)
        crisis_signal = False
        grief_orientation = analysis.orientation
    crisis_escalated = crisis_score >= CRISIS_ESCALATION_THRESHOLD

    history = []
    if not crisis_escalated:
        # 3. Загрузить историю до сохранения текущего сообщения
        with SessionLocal() as s:
            history_rows = s.execute(
                select(ChatMessage)
                .where(ChatMessage.twin_id == twin_id, ChatMessage.user_id == user_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(20)
            ).scalars().all()
        history = [
            {"role": r.role, "content": r.content}
            for r in reversed(history_rows)
        ]

    with SessionLocal() as s:
        s.add(
            ChatMessage(
                twin_id=twin_id,
                user_id=user_id,
                role="user",
                content=user_msg,
                grief_orientation=grief_orientation,
                crisis_signal=crisis_signal,
                crisis_score=crisis_score,
                crisis_escalated=crisis_escalated,
            )
        )
        s.commit()

    if crisis_escalated:
        answer = CRISIS_RESOURCES
    else:
        # 4. RAG-поиск + промпт с учётом состояния
        context = retrieve(user_msg, twin_id)
        system = build_system_prompt(
            twin_id,
            state=current_state,
            grief_orientation=grief_orientation,
            crisis_signal=bool(crisis_signal),
        )
        if context:
            system += f"\n\nРелевантные воспоминания:\n{context}"

        # 5. Вызов модели
        messages = [{"role": "system", "content": system}] + history + [
            {"role": "user", "content": user_msg}
        ]
        resp = get_ollama_client().chat(
            model=CHAT_MODEL,
            messages=messages,
            think=False,
            options={"num_ctx": 8192},
        )
        answer = resp["message"]["content"]

    # 6. Сохранить ответ ассистента
    with SessionLocal() as s:
        s.add(ChatMessage(twin_id=twin_id, user_id=user_id, role="assistant", content=answer))
        s.commit()

    return answer