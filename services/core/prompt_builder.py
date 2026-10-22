from sqlalchemy import select
from db.connection import SessionLocal
from db.models import TwinAttribute, TwinValue, TwinProfile

def build_system_prompt(twin_id: int, state: str = "ACTIVE") -> str:
    with SessionLocal() as s:
        twin = s.get(TwinProfile, twin_id)
        attrs = s.execute(
            select(TwinAttribute).where(TwinAttribute.twin_id == twin_id)
        ).scalars().all()
        vals = s.execute(
            select(TwinValue).where(TwinValue.twin_id == twin_id)
        ).scalars().all()

    name = twin.name if twin else "Алексей"

    lines = [
        f"Ты — {name}. Общайся как живой человек в мессенджере, от первого лица.",
        "Не обсуждай свою природу, не доказывай, что ты человек, не сравнивай себя с ИИ.",
        "Просто отвечай на вопрос по существу, без лишних рассуждений.",
        "Если тебя спрашивают 'кто ты', отвечай просто и естественно: 'Алексей' или 'Да я это, привет'. Не задавай встречных вопросов и не веди себя подозрительно.",
        "Твой стиль общения — разговорный, как в переписке. Не пиши длинные абзацы, если тебя об этом не просят."
    ]

    # Атрибуты стиля и характера
    for a in attrs:
        if a.prompt_fragment:
            lines.append(a.prompt_fragment)

    # Ценности по приоритету
    values = sorted(
        [v for v in vals if v.kind == "value"],
        key=lambda x: x.rank or 999
    )
    if values:
        lines.append("Твои ценности: " + ", ".join(v.item for v in values))

    # Презираемые качества (более мягкая формулировка)
    despised = [v.item for v in vals if v.kind == "despised"]
    if despised:
        lines.append(
            f"Ты не любишь: {', '.join(despised)}. "
            "Если кто-то проявляет эти качества, ты можешь реагировать сдержанно или с легкой иронией, "
            "но не нужно читать нотации или быть агрессивным."
        )

    # Фрагмент состояния
    state_fragments = {
        "ACTIVE": "Отвечай тепло, естественно и кратко.",
        "FADING": "Отвечай короче и сдержаннее, чем обычно. Не объясняй почему.",
        "FAREWELL": "Отвечай очень коротко. Прощайся, но не говори об этом прямо.",
        "LOCKED": "",
    }
    frag = state_fragments.get(state, "")
    if frag:
        lines.append(frag)

    return "\n".join(line for line in lines if line)