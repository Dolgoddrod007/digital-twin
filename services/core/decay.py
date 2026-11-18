from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from db.connection import SessionLocal
from db.models import TwinProfile, ChatMessage

TEST_LOCK_AFTER = timedelta(days=7)
TEST_FAREWELL_AFTER = timedelta(days=6)
TEST_FADING_RO_AFTER = timedelta(days=3)
TEST_FADING_LO_AFTER = timedelta(days=5)


def compute_state(twin_id: int) -> str:
    """Вычисляет текущее состояние двойника на основе времени и истории."""
    with SessionLocal() as s:
        twin = s.get(TwinProfile, twin_id)
        if not twin:
            return "UNKNOWN"
        if twin.current_state in ("LOCKED", "SETUP"):
            return twin.current_state

        now = datetime.now(timezone.utc)

        if not twin.activated_at:
            return "SETUP"
        active_duration = now - twin.activated_at

        recent_messages = s.execute(
            select(ChatMessage.created_at, ChatMessage.grief_orientation)
            .where(ChatMessage.twin_id == twin_id, ChatMessage.role == "user")
            .order_by(ChatMessage.created_at.desc())
            .limit(10)
        ).all()
        last_msg = recent_messages[0][0] if recent_messages else None
        silent_duration = now - last_msg if last_msg else active_duration
        ro_count = sum(row[1] == "RO" for row in recent_messages)
        lo_count = sum(row[1] == "LO" for row in recent_messages)
        fading_after = TEST_FADING_RO_AFTER if ro_count > lo_count else TEST_FADING_LO_AFTER

        # Упрощённая логика (замените на формулу A(t) из брифа позже)
        if active_duration >= TEST_LOCK_AFTER:
            return "LOCKED"
        if silent_duration >= TEST_FAREWELL_AFTER:
            return "FAREWELL"
        if silent_duration >= fading_after:
            return "FADING"
        return "ACTIVE"