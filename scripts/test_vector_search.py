import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from db.connection import SessionLocal
from db.models import TwinMemory
from services.core.ollama_client import get_ollama_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only проверка векторного поиска по существующему профилю"
    )
    parser.add_argument("--twin-id", required=True, type=int)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    vector = get_ollama_client().embeddings(
        model=model,
        prompt=args.query,
    )["embedding"]
    with SessionLocal() as session:
        rows = session.execute(
            select(
                TwinMemory.id,
                TwinMemory.embedding.cosine_distance(vector).label("distance"),
                TwinMemory.source_type,
                TwinMemory.source_author,
                TwinMemory.source_date,
            )
            .where(
                TwinMemory.twin_id == args.twin_id,
                TwinMemory.embedding.is_not(None),
                TwinMemory.embedding_model == model,
            )
            .order_by(TwinMemory.embedding.cosine_distance(vector))
            .limit(max(1, args.limit))
        ).all()

    print(f"Модель: {model}; размерность запроса: {len(vector)}; найдено: {len(rows)}")
    for row in rows:
        print(
            f"id={row.id}; similarity={1.0 - float(row.distance):.3f}; "
            f"type={row.source_type or 'unknown'}; "
            f"origin={row.source_author or 'unknown'}; "
            f"date={row.source_date or 'unknown'}"
        )
    print("Тексты воспоминаний не выводились; БД не изменялась.")


if __name__ == "__main__":
    main()