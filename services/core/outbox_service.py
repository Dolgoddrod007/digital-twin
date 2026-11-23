from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, or_, select

from db.connection import SessionLocal
from db.models import OutboxMessage

CLAIM_TIMEOUT = timedelta(minutes=2)
MAX_DELIVERY_ATTEMPTS = 8


@dataclass(frozen=True)
class OutboxDelivery:
    id: int
    recipient_id: str
    content: str
    telegram_placeholder_message_id: int | None


def claim_next_delivery(channel: str) -> OutboxDelivery | None:
    now = datetime.now(timezone.utc)
    stale_claim = now - CLAIM_TIMEOUT
    with SessionLocal() as session:
        message = session.execute(
            select(OutboxMessage)
            .where(
                OutboxMessage.channel == channel,
                or_(
                    and_(
                        OutboxMessage.status == "pending",
                        OutboxMessage.deliver_at <= now,
                    ),
                    and_(
                        OutboxMessage.status == "sending",
                        or_(
                            OutboxMessage.claimed_at.is_(None),
                            OutboxMessage.claimed_at <= stale_claim,
                        ),
                    ),
                ),
            )
            .order_by(OutboxMessage.deliver_at, OutboxMessage.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if message is None:
            return None

        message.status = "sending"
        message.claimed_at = now
        message.attempts += 1
        delivery = OutboxDelivery(
            id=message.id,
            recipient_id=message.recipient_id,
            content=message.content,
            telegram_placeholder_message_id=message.telegram_placeholder_message_id,
        )
        session.commit()
        return delivery


def mark_delivery_sent(delivery_id: int) -> None:
    with SessionLocal() as session:
        message = session.get(OutboxMessage, delivery_id)
        if message is None:
            raise LookupError(f"Outbox message {delivery_id} was not found")
        message.status = "sent"
        message.sent_at = datetime.now(timezone.utc)
        message.claimed_at = None
        message.last_error = None
        session.commit()


def mark_delivery_failed(delivery_id: int, error_type: str) -> bool:
    with SessionLocal() as session:
        message = session.get(OutboxMessage, delivery_id)
        if message is None:
            raise LookupError(f"Outbox message {delivery_id} was not found")

        message.last_error = error_type
        message.claimed_at = None
        if message.attempts >= MAX_DELIVERY_ATTEMPTS:
            message.status = "failed"
            session.commit()
            return False

        message.status = "pending"
        retry_delay = min(2 ** message.attempts, 60)
        message.deliver_at = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
        session.commit()
        return True
