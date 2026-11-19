from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from db.connection import SessionLocal
from db.models import TwinProfile, ChatMessage

TEST_LOCK_AFTER = timedelta(days=7)
TEST_FAREWELL_AFTER = timedelta(days=6)
TEST_FADING_RO_AFTER = timedelta(days=3)
TEST_FADING_LO_AFTER = timedelta(days=5)
AVAILABILITY_WINDOW = timedelta(days=3)
ALPHA = 0.3
BETA = 0.2


def _update_effective_availability(
    session: Session,
    twin: TwinProfile,
    now: datetime,
    activated_at: datetime,
    active_duration: timedelta,
) -> float:
    schedule_days = TEST_LOCK_AFTER.total_seconds() / 86400
    elapsed_position = min(
        max(active_duration.total_seconds() / 86400, 0.0),
        schedule_days,
    )
    previous_position = twin.schedule_position or 0.0
    schedule_position = min(schedule_days, max(previous_position, elapsed_position))
    scheduled_availability = max(0.0, 1.0 - schedule_position / schedule_days)
    window_start = max(now - AVAILABILITY_WINDOW, activated_at)
    orientations = session.execute(
        select(ChatMessage.grief_orientation)
        .where(
            ChatMessage.twin_id == twin.id,
            ChatMessage.role == "user",
            ChatMessage.created_at >= window_start,
            ChatMessage.grief_orientation.in_(("LO", "RO")),
        )
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
    ).scalars().all()

    lo_weight = orientations.count("LO") / len(orientations) if orientations else 0.0
    transitions = sum(
        previous != current
        for previous, current in zip(orientations, orientations[1:])
    )
    oscillation = transitions / (len(orientations) - 1) if len(orientations) > 1 else 0.0
    current_availability = (
        scheduled_availability
        * (1.0 - ALPHA * lo_weight)
        * (1.0 - BETA * oscillation)
    )
    previous_availability = twin.availability_effective
    if previous_availability is None:
        previous_availability = 1.0

    effective_availability = min(
        max(0.0, min(1.0, previous_availability)),
        max(0.0, min(1.0, current_availability)),
    )
    twin.schedule_position = schedule_position
    twin.availability_effective = effective_availability
    return effective_availability


def compute_state(twin_id: int) -> str:
    """Вычисляет текущее состояние двойника на основе времени и истории."""
    with SessionLocal() as s:
        twin = s.execute(
            select(TwinProfile)
            .where(TwinProfile.id == twin_id)
            .with_for_update()
        ).scalar_one_or_none()
        if not twin:
            return "UNKNOWN"
        if twin.current_state == "LOCKED":
            twin.availability_effective = 0.0
            s.commit()
            return "LOCKED"
        if twin.current_state == "SETUP":
            return twin.current_state

        now = datetime.now(timezone.utc)

        activated_at = twin.activated_at
        if not activated_at:
            return "SETUP"
        active_duration = now - activated_at
        _update_effective_availability(
            s,
            twin,
            now,
            activated_at,
            active_duration,
        )

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

        if active_duration >= TEST_LOCK_AFTER:
            state = "LOCKED"
            twin.availability_effective = 0.0
        elif silent_duration >= TEST_FAREWELL_AFTER:
            state = "FAREWELL"
        elif silent_duration >= fading_after:
            state = "FADING"
        else:
            state = "ACTIVE"
        s.commit()
        return state