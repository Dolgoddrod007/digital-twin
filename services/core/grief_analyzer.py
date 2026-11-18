import math
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

from services.core.ollama_client import get_ollama_client

GriefOrientation = Literal["LO", "RO"]
LOSS_ORIENTED_REFERENCE = (
    "Я сосредоточен на утрате: скучаю по умершему близкому, вспоминаю его, "
    "переживаю грусть, тоску, вину или пустоту и пытаюсь осмыслить потерю."
)
RESTORATION_ORIENTED_REFERENCE = (
    "Я постепенно возвращаюсь к повседневной жизни: занимаюсь работой и делами, "
    "общаюсь, строю планы, осваиваю новые занятия и замечаю моменты интереса и радости, "
    "хотя всё ещё могу скучать по близкому."
)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must have the same non-zero length")

    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding vectors must have non-zero magnitude")

    similarity = math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    similarity = min(1.0, max(-1.0, similarity))
    return 1.0 - similarity


@lru_cache(maxsize=1)
def _reference_embeddings(model: str) -> tuple[list[float], list[float]]:
    client = get_ollama_client()
    loss_embedding = client.embeddings(
        model=model,
        prompt=LOSS_ORIENTED_REFERENCE,
    )["embedding"]
    restoration_embedding = client.embeddings(
        model=model,
        prompt=RESTORATION_ORIENTED_REFERENCE,
    )["embedding"]
    return loss_embedding, restoration_embedding


def analyze_grief_orientation(text: str) -> GriefOrientation:
    content = text.strip()
    if not content:
        raise ValueError("Cannot classify an empty message")

    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    client = get_ollama_client()
    message_embedding = client.embeddings(model=model, prompt=content)["embedding"]
    loss_embedding, restoration_embedding = _reference_embeddings(model)

    if _cosine_distance(message_embedding, loss_embedding) < _cosine_distance(
        message_embedding,
        restoration_embedding,
    ):
        return "LO"
    return "RO"
