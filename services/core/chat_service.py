from datetime import datetime, timezone
import ollama
from sqlalchemy import select
from db.connection import SessionLocal
from db.models import TwinProfile, TwinMemory, ChatMessage
from services.core.prompt_builder import build_system_prompt
from services.core.decay import compute_state

EMBED_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:8b"

def retrieve(query: str, twin_id: int, k: int = 3) -> str:
    q = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    with SessionLocal() as s:
        rows = s.execute(
            select(TwinMemory.content)
            .where(TwinMemory.twin_id == twin_id)
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

    # 3. Загрузить историю из БД
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

    # 4. RAG-поиск + промпт с учётом состояния
    context = retrieve(user_msg, twin_id)
    system = build_system_prompt(twin_id, state=current_state)
    if context:
        system += f"\n\nРелевантные воспоминания:\n{context}"

    # 5. Вызов модели
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": user_msg}
    ]
    resp = ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_ctx": 8192})
    answer = resp["message"]["content"]

    # 6. Сохранить оба сообщения
    with SessionLocal() as s:
        s.add(ChatMessage(twin_id=twin_id, user_id=user_id, role="user", content=user_msg))
        s.add(ChatMessage(twin_id=twin_id, user_id=user_id, role="assistant", content=answer))
        s.commit()

    return answer