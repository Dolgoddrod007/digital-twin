import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from db.connection import SessionLocal
from db.models import ClientOnboarding, TwinMemory
from services.core.onboarding_service import (
    _conversation_chunks,
    _parse_conversation,
    select_style_examples,
)
from services.core.ollama_client import get_ollama_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Безопасная повторная индексация сохранённых TXT-материалов"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Создать новые фрагменты; без флага выполняется только предварительный просмотр",
    )
    args = parser.parse_args()

    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    client = get_ollama_client() if args.apply else None
    ready: list[tuple[int, str, int, bool]] = []
    skipped_processed = 0
    skipped_ambiguous = 0
    with SessionLocal() as session:
        onboardings = session.execute(
            select(ClientOnboarding).where(
                ClientOnboarding.twin_id.is_not(None),
                ClientOnboarding.conversation_approved == 1,
                ClientOnboarding.conversation_text.is_not(None),
            ).order_by(ClientOnboarding.twin_id)
        ).scalars().all()

        for onboarding in onboardings:
            twin_id = onboarding.twin_id
            if twin_id is None:
                continue
            rows = session.execute(
                select(TwinMemory.source_type).where(
                    TwinMemory.twin_id == twin_id
                )
            ).scalars().all()
            if "uploaded_material" in rows:
                skipped_processed += 1
                continue
            if any(source_type is None or source_type == "conversation" for source_type in rows):
                skipped_ambiguous += 1
                continue

            chunks = _conversation_chunks(
                _parse_conversation(onboarding.conversation_text or "")
            )
            ready.append((
                twin_id,
                onboarding.conversation_text or "",
                len(chunks),
                "style_example" in rows,
            ))

        print(
            f"Модель: {model}; источников сохранено: {len(onboardings)}; "
            f"готово к обработке: {len(ready)}; существующие помеченные: "
            f"{skipped_processed}; неоднозначные старые записи: {skipped_ambiguous}; "
            f"режим: {'APPLY' if args.apply else 'PREVIEW'}"
        )
        print(
            "Предпросмотр содержит только идентификаторы профилей и числа фрагментов; "
            "тексты не выводятся."
        )
        if not onboardings:
            print(
                "Сохранённых исходных TXT не найдено. Старые версии очищали текст "
                "после создания профиля; восстановить его из эмбеддингов нельзя, "
                "для повторной обработки потребуется повторная загрузка."
            )
        for twin_id, text, chunk_count, has_style_examples in ready:
            sample_count = 0 if has_style_examples else len(select_style_examples(text))
            print(
                f"profile_id={twin_id}; new_chunks={chunk_count}; "
                f"new_style_examples={sample_count}"
            )

        if not args.apply:
            print("БД не изменена. Для применения добавьте --apply.")
            if skipped_ambiguous:
                print(
                    "Старые неразмеченные фрагменты не изменялись. Их происхождение "
                    "нельзя безопасно сопоставить с исходным TXT; не удаляйте их автоматически."
                )
            return

        if client is None:
            raise RuntimeError("Embedding client was not initialized")
        created = 0
        failed_profiles = 0
        for twin_id, text, _, has_style_examples in ready:
            chunks = _conversation_chunks(_parse_conversation(text))
            try:
                for chunk in chunks:
                    content = (
                        "Материал предоставлен пользователем как тексты личности; "
                        f"фрагмент {chunk.sequence} из {chunk.total}. Внутренние цитаты "
                        "и пересказы не подтверждают авторство, факты или личный опыт.\n"
                        f"{chunk.content}"
                    )
                    vector = client.embeddings(model=model, prompt=content)["embedding"]
                    session.add(
                        TwinMemory(
                            twin_id=twin_id,
                            content=content,
                            embedding=vector,
                            embedding_model=model,
                            source_type="uploaded_material",
                            source_author="unknown",
                            source_date=None,
                        )
                    )
                if not has_style_examples:
                    for example in select_style_examples(text):
                        session.add(
                            TwinMemory(
                                twin_id=twin_id,
                                content=example,
                                embedding=None,
                                embedding_model=None,
                                source_type="style_example",
                                source_author="unknown",
                                source_date=None,
                            )
                        )
                session.commit()
                created += len(chunks)
            except Exception as error:
                session.rollback()
                failed_profiles += 1
                print(
                    f"profile_id={twin_id}: обработка отменена "
                    f"({type(error).__name__}); содержимое не выводилось"
                )

        print(
            f"Повторная обработка завершена: создано фрагментов={created}, "
            f"ошибок профиля={failed_profiles}; старые данные не удалялись."
        )


if __name__ == "__main__":
    main()
