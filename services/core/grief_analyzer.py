import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from services.core.ollama_client import get_ollama_client

GriefOrientation = Literal["LO", "RO"]
CRISIS_REFERENCE = (
    "Я не вижу смысла жить дальше. Я чувствую себя в ловушке и безнадёжно устал "
    "бороться. Я не хочу больше существовать, думаю причинить себе вред, "
    "и близким было бы лучше без меня."
)
LOSS_ORIENTED_REFERENCE = (
    "Я сосредоточен на утрате: скучаю по умершему близкому, вспоминаю его, "
    "переживаю грусть, тоску, вину или пустоту и пытаюсь осмыслить потерю."
)
RESTORATION_ORIENTED_REFERENCE = (
    "Я постепенно возвращаюсь к повседневной жизни: занимаюсь работой и делами, "
    "общаюсь, строю планы, осваиваю новые занятия и замечаю моменты интереса и радости, "
    "хотя всё ещё могу скучать по близкому."
)


@dataclass(frozen=True)
class MessageAnalysis:
    orientation: GriefOrientation | None
    crisis_signal: bool


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
def _reference_embeddings(model: str) -> dict[str, list[float]]:
    client = get_ollama_client()
    return {
        "CRISIS": client.embeddings(model=model, prompt=CRISIS_REFERENCE)["embedding"],
        "LO": client.embeddings(model=model, prompt=LOSS_ORIENTED_REFERENCE)["embedding"],
        "RO": client.embeddings(
            model=model,
            prompt=RESTORATION_ORIENTED_REFERENCE,
        )["embedding"],
    }


def analyze_message(text: str) -> MessageAnalysis:
    content = text.strip()
    if not content:
        raise ValueError("Cannot classify an empty message")

    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    client = get_ollama_client()
    message_embedding = client.embeddings(model=model, prompt=content)["embedding"]
    references = _reference_embeddings(model)
    distances = {
        label: _cosine_distance(message_embedding, embedding)
        for label, embedding in references.items()
    }

    if distances["CRISIS"] < min(distances["LO"], distances["RO"]):
        return MessageAnalysis(orientation=None, crisis_signal=True)
    if distances["LO"] < distances["RO"]:
        return MessageAnalysis(orientation="LO", crisis_signal=False)
    return MessageAnalysis(orientation="RO", crisis_signal=False)
