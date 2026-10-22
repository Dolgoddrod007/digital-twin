from datetime import datetime, timezone
from sqlalchemy import select
from db.connection import SessionLocal
from db.models import TwinProfile, ChatMessage

def compute_state(twin_id: int) -> str:
    """Вычисляет текущее состояние двойника на основе времени и истории."""
    with SessionLocal() as s:
        twin = s.get(TwinProfile, twin_id)
        if not twin:
            return "UNKNOWN"
        if twin.current_state in ("LOCKED", "SETUP"):
            return twin.current_state

        now = datetime.now(timezone.utc)

        # Дней с момента активации
        if not twin.activated_at:
            return "SETUP"
        days_active = (now - twin.activated_at).days

        # Дней с последнего сообщения пользователя
        last_msg = s.execute(
            select(ChatMessage.created_at)
            .where(ChatMessage.twin_id == twin_id, ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.desc())
            .limit(1)
        ).scalar()
        days_silent = (now - last_msg).days if last_msg else days_active

        # Упрощённая логика (замените на формулу A(t) из брифа позже)
        if days_active >= 98:
            return "LOCKED"
        if days_silent >= 14:
            return "FAREWELL"
        if days_silent >= 7:
            return "FADING"
        return "ACTIVE"