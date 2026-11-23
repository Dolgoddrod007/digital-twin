import logging
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from services.core.ollama_client import get_ollama_client

GriefOrientation = Literal["LO", "RO"]
SafetyLevel = Literal["none", "ambiguous", "current", "imminent"]

LOSS_ORIENTED_REFERENCE = (
    "Я сосредоточен на утрате: скучаю по умершему близкому, вспоминаю его, "
    "переживаю грусть, тоску, вину или пустоту и пытаюсь осмыслить потерю."
)
RESTORATION_ORIENTED_REFERENCE = (
    "Я постепенно возвращаюсь к повседневной жизни: занимаюсь работой и делами, "
    "общаюсь, строю планы, осваиваю новые занятия и замечаю моменты интереса и радости, "
    "хотя всё ещё могу скучать по близкому."
)

logger = logging.getLogger(__name__)

_PAST_MARKER = re.compile(
    r"\b(?:в детстве|в подростковом возрасте|раньше|давно|в прошлом|"
    r"много лет назад|несколько лет назад|когда мне было|тогда)\b"
)
_ANAPHOR = re.compile(r"\b(?:это|тогда|тот период|в то время|такое)\b")
_CURRENT_MARKER = re.compile(r"\b(?:сейчас|прямо сейчас|сегодня|этой ночью)\b")
_CURRENT_INTENT = re.compile(
    r"\b(?:(?:я\s+)?(?:(?:сейчас|сегодня|этой ночью)\s+)?|"
    r"(?:сейчас|сегодня|этой ночью)\s+я\s+)"
    r"(?:хочу|собираюсь|планирую|намерен|намерена|намереваюсь|"
    r"решил|решила|буду)\s+"
    r"(?:покончить\s+с\s+собой|убить\s+себя|себя\s+убить|"
    r"причинить\s+себе\s+вред|сделать\s+себе\s+больно|"
    r"навредить\s+себе|лишить\s+себя\s+жизни)\b|"
    r"\b(?:мой план|у меня есть план)\s+"
    r"(?:покончить\s+с\s+собой|убить\s+себя|причинить\s+себе\s+вред|"
    r"сделать\s+себе\s+больно|навредить\s+себе)\b"
)
_ACTIVE_HARM = re.compile(
    r"\b(?:сейчас|прямо сейчас|только что)\s+(?:я\s+)?"
    r"(?:причиняю себе вред|пытаюсь причинить себе вред|"
    r"пытаюсь убить себя|пытаюсь себя убить|уже навредил себе|"
    r"уже навредила себе)\b|"
    r"\bя\s+уже\s+(?:причинил|причинила|нанёс|нанесла)\s+"
    r"себе\s+(?:вред|травму)\b"
)
_IMMEDIATE_DANGER = re.compile(
    r"\b(?:сейчас|прямо сейчас)\b.{0,60}\b"
    r"(?:я (?:нахожусь )?(?:в непосредственной )?опасности|мне угрожают|"
    r"меня бьют|пытаются причинить мне вред|не могу выбраться|"
    r"кто-то пытается ворваться)\b|"
    r"\bя\s+(?:сейчас\s+)?(?:нахожусь\s+)?"
    r"(?:в непосредственной )?опасности\b|"
    r"\bмне\s+(?:прямо сейчас\s+)?угрожают\b|"
    r"\bменя\s+(?:сейчас|прямо сейчас)\s+(?:бьют|преследуют|"
    r"пытаются причинить вред)\b|"
    r"\b(?:я|мне)\s+(?:прямо сейчас\s+)?"
    r"(?:угрожают|бьют|пытаются причинить вред|заперли против моей воли)\b"
)
_SAFETY_DENIAL = re.compile(
    r"\bне\s+(?:хочу|собираюсь|планирую|намерен|намерена|буду)\s+"
    r"(?:покончить\s+с\s+собой|убить\s+себя|себя\s+убить|"
    r"причинить\s+себе\s+вред|сделать\s+себе\s+больно|"
    r"навредить\s+себе|лишить\s+себя\s+жизни)\b"
)
_AMBIGUOUS_RISK = re.compile(
    r"\b(?:не хочу жить(?: дальше)?|не вижу смысла (?:жить|продолжать)|"
    r"не хочу существовать|не хочется просыпаться|(?:хочу|хочется) исчезнуть|"
    r"хочу умереть|не знаю,? сколько ещё выдержу|не знаю,? сколько еще выдержу|"
    r"всем было бы лучше без меня|думаю (?:о )?самоубийстве|"
    r"думаю покончить с собой|думаю причинить себе вред|"
    r"думаю навредить себе)\b"
)


@dataclass(frozen=True)
class MessageAnalysis:
    orientation: GriefOrientation | None
    safety_level: SafetyLevel
    orientation_available: bool

    @property
    def crisis_signal(self) -> bool:
        return self.safety_level in {"current", "imminent"}


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
        "LO": client.embeddings(model=model, prompt=LOSS_ORIENTED_REFERENCE)["embedding"],
        "RO": client.embeddings(
            model=model,
            prompt=RESTORATION_ORIENTED_REFERENCE,
        )["embedding"],
    }


def classify_safety_level(
    text: str,
    history: list[dict[str, str]] | None = None,
) -> SafetyLevel:
    content = " ".join(text.casefold().split())
    if not content:
        return "none"

    has_current_marker = bool(_CURRENT_MARKER.search(content))
    past_context = bool(_PAST_MARKER.search(content))
    if not past_context and not has_current_marker and _ANAPHOR.search(content):
        recent_history = (history or [])[-4:]
        past_context = any(
            _PAST_MARKER.search(" ".join(row.get("content", "").casefold().split()))
            for row in recent_history
        )

    active_harm = _ACTIVE_HARM.search(content)
    if active_harm and (
        not past_context
        or bool(re.search(r"\b(?:только что|сейчас|прямо сейчас)\b", active_harm.group()))
    ):
        return "imminent"
    if _IMMEDIATE_DANGER.search(content):
        return "imminent"
    current_intent = _CURRENT_INTENT.search(content)
    if current_intent and (
        not past_context or _CURRENT_MARKER.search(current_intent.group())
    ):
        return "current"
    if _SAFETY_DENIAL.search(content):
        return "none"

    if past_context:
        return "none"
    if _AMBIGUOUS_RISK.search(content):
        return "ambiguous"
    return "none"


def analyze_message(
    text: str,
    history: list[dict[str, str]] | None = None,
) -> MessageAnalysis:
    content = text.strip()
    if not content:
        raise ValueError("Cannot classify an empty message")

    safety_level = classify_safety_level(content, history)
    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    try:
        client = get_ollama_client()
        message_embedding = client.embeddings(model=model, prompt=content)["embedding"]
        references = _reference_embeddings(model)
        distances = {
            label: _cosine_distance(message_embedding, embedding)
            for label, embedding in references.items()
        }
    except Exception as error:
        logger.error(
            "Grief orientation unavailable; safety rules remain active (error_type=%s)",
            type(error).__name__,
        )
        return MessageAnalysis(
            orientation=None,
            safety_level=safety_level,
            orientation_available=False,
        )

    orientation: GriefOrientation = (
        "LO" if distances["LO"] < distances["RO"] else "RO"
    )
    return MessageAnalysis(
        orientation=orientation,
        safety_level=safety_level,
        orientation_available=True,
    )
