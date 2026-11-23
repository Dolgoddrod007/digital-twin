import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import or_, select

from db.connection import SessionLocal
from db.models import TwinMemory
from services.core.ollama_client import get_ollama_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Переиндексация воспоминаний текущей моделью Ollama"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Записать новые эмбеддинги; без флага выполняется только предварительный просмотр",
    )
    args = parser.parse_args()

    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    with SessionLocal() as session:
        memories = session.execute(
            select(TwinMemory).where(
                or_(
                    TwinMemory.embedding.is_(None),
                    TwinMemory.embedding_model.is_(None),
                    TwinMemory.embedding_model != model,
                ),
                or_(
                    TwinMemory.source_type.is_(None),
                    ~TwinMemory.source_type.in_(
                        ("style_example", "style_example_archived")
                    ),
                )
            ).order_by(TwinMemory.id)
        ).scalars().all()
        print(
            f"Модель: {model}; записей, которые будут обновлены: {len(memories)}; "
            f"режим: {'APPLY' if args.apply else 'PREVIEW'}"
        )
        if not args.apply:
            print("Данные БД не изменены. Для запуска добавьте --apply.")
            return

        client = get_ollama_client()
        reindexed = 0
        failed = 0
        for memory in memories:
            try:
                vector = client.embeddings(model=model, prompt=memory.content)["embedding"]
            except Exception as error:
                failed += 1
                print(
                    f"Запись #{memory.id}: ошибка эмбеддинга ({type(error).__name__}); "
                    "текст не выводился"
                )
                continue
            memory.embedding = vector
            memory.embedding_model = model
            reindexed += 1
        session.commit()

        print(
            f"Переиндексация завершена: обновлено={reindexed}, ошибок={failed}"
        )


if __name__ == "__main__":
    main()
