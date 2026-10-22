import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ollama
from db.connection import SessionLocal
from db.models import TwinProfile, TwinMemory
from sqlalchemy import select

# 0. Убедитесь, что модель эмбеддингов скачана: ollama pull qwen3-embedding:0.6b
EMBED_MODEL = "qwen3-embedding:0.6b"

# 1. Создать тестового двойника
with SessionLocal() as s:
    twin = TwinProfile(name="test_twin")
    s.add(twin)
    s.commit()
    s.refresh(twin)
    twin_id = twin.id

# 2. Создать эмбеддинг
resp = ollama.embeddings(model=EMBED_MODEL, prompt="Я боюсь остаться один")
vec = resp["embedding"]
print(f"Размерность вектора: {len(vec)}")  # должно быть 1024

# 3. Записать в базу данных
with SessionLocal() as s:
    mem = TwinMemory(twin_id=twin_id, content="Я боюсь остаться один", embedding=vec)
    s.add(mem)
    s.commit()

# 4. Поиск по сходству
q = ollama.embeddings(model=EMBED_MODEL, prompt="одиночество")["embedding"]
with SessionLocal() as s:
    rows = s.execute(
        select(TwinMemory.content)
        .where(TwinMemory.twin_id == twin_id)
        .order_by(TwinMemory.embedding.cosine_distance(q))
        .limit(1)
    ).all()
    print(rows)