from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session
from db.connection import SessionLocal
from db.models import TwinProfile, ChatMessage

TEST_LOCK_AFTER = timedelta(days=7)
AVAILABILITY_WINDOW = timedelta(days=3)
ALPHA = 0.3
BETA = 0.2
ACTIVE_AVAILABILITY_THRESHOLD = 0.7
ACTIVE_RECOVERY_THRESHOLD = 0.75
FAREWELL_AVAILABILITY_THRESHOLD = 0.35


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

    lo_weight = (
        orientations.count("LO") / len(orientations)
        if len(orientations) >= 3
        else 0.5
    )
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
    effective_availability = max(0.0, min(1.0, current_availability))
    twin.schedule_position = schedule_position
    twin.availability_effective = effective_availability
    return effective_availability


def _state_for_availability(availability: float, previous_state: str | None) -> str:
    if previous_state == "LOCKED":
        return "LOCKED"
    if previous_state == "FAREWELL":
        return "FAREWELL"
    if availability < FAREWELL_AVAILABILITY_THRESHOLD:
        return "FAREWELL"
    if previous_state == "FADING":
        return (
            "ACTIVE"
            if availability > ACTIVE_RECOVERY_THRESHOLD
            else "FADING"
        )
    if availability <= ACTIVE_AVAILABILITY_THRESHOLD:
        return "FADING"
    return "ACTIVE"


def compute_state(twin_id: int) -> str:
    """Recalculate availability and state, respecting farewell and lifetime locks."""
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
        availability = _update_effective_availability(
            s,
            twin,
            now,
            activated_at,
            active_duration,
        )

        if active_duration >= TEST_LOCK_AFTER:
            state = "LOCKED"
            twin.availability_effective = 0.0
        else:
            state = _state_for_availability(availability, twin.current_state)
        twin.current_state = state
        s.commit()
        return state