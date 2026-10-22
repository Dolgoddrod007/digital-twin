from datetime import datetime, timezone
from db.connection import SessionLocal
from db.models import TwinProfile, TwinAttribute, TwinValue, TwinMemory
import ollama

with SessionLocal() as s:
    twin = TwinProfile(
        name="Алексей",
        status="active",
        current_state="ACTIVE",
        activated_at=datetime.now(timezone.utc),
    )
    s.add(twin)
    s.commit()
    s.refresh(twin)
    tid = twin.id

    # Стиль
    s.add(TwinAttribute(twin_id=tid, block="style", attribute_key="speech_length",
        raw_answer="коротко и по существу", prompt_fragment="Отвечай коротко, без лишних деталей."))
    s.add(TwinAttribute(twin_id=tid, block="style", attribute_key="humor_type",
        raw_answer="тонкий сарказм", prompt_fragment="Шути с тонким сарказмом, но не грубо."))
    # ... остальные атрибуты

    # Ценности
    s.add(TwinValue(twin_id=tid, kind="value", item="верность", rank=1))
    s.add(TwinValue(twin_id=tid, kind="value", item="семья", rank=2))
    s.add(TwinValue(twin_id=tid, kind="despised", item="предательство", rank=None))

    # Память (с эмбеддингом)
    text = "В юности потерял близкого друга и с тех пор ценит честность."
    vec = ollama.embeddings(model="qwen3-embedding:0.6b", prompt=text)["embedding"]
    s.add(TwinMemory(twin_id=tid, content=text, embedding=vec))

    s.commit()
    print(f"Персонаж создан, id={tid}")