import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from db.connection import SessionLocal
from db.models import ClientOnboarding, TwinMemory
from services.core.onboarding_service import load_conversation_text, select_style_examples


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Просмотр и безопасная замена примеров речевой манеры профиля"
    )
    parser.add_argument("--twin-id", required=True, type=int)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--show",
        action="store_true",
        help="Явно вывести активные примеры в локальный терминал",
    )
    action.add_argument(
        "--replace-from",
        type=Path,
        help="Подготовить новые примеры из локального TXT-файла",
    )
    action.add_argument(
        "--use-saved-source",
        action="store_true",
        help="Подготовить примеры из сохранённого исходного TXT профиля",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Применить замену; без флага выводится только число выбранных примеров",
    )
    args = parser.parse_args()

    with SessionLocal() as session:
        active = session.execute(
            select(TwinMemory)
            .where(
                TwinMemory.twin_id == args.twin_id,
                TwinMemory.source_type == "style_example",
            )
            .order_by(TwinMemory.id)
        ).scalars().all()

        if args.show:
            if not active:
                print("У профиля нет активных примеров речевой манеры.")
                return
            for index, memory in enumerate(active, start=1):
                print(f"[{index}] {memory.content}")
            print("Тексты показаны только по явному запросу --show.")
            return

        if args.replace_from:
            source_text = load_conversation_text(args.replace_from)
            source_description = "локальный исходный TXT"
        else:
            source_text = session.execute(
                select(ClientOnboarding.conversation_text)
                .where(
                    ClientOnboarding.twin_id == args.twin_id,
                    ClientOnboarding.conversation_approved == 1,
                    ClientOnboarding.conversation_text.is_not(None),
                )
                .limit(1)
            ).scalar_one_or_none()
            if not source_text:
                fragments = session.execute(
                    select(TwinMemory.content)
                    .where(
                        TwinMemory.twin_id == args.twin_id,
                        TwinMemory.source_type == "uploaded_material",
                    )
                    .order_by(TwinMemory.id)
                ).scalars().all()
                if not fragments:
                    raise ValueError(
                        "Для профиля нет сохранённого исходного TXT или TXT-фрагментов; "
                        "укажите --replace-from с локальной копией файла."
                    )
                source_text = "\n\n".join(fragments)
                source_description = (
                    "сохранённые TXT-фрагменты; исходный файл недоступен, "
                    "примеры не восстановлены из эмбеддингов"
                )
            else:
                source_description = "сохранённый исходный TXT"

        examples = select_style_examples(source_text)
        print(
            f"Источник: {source_description}; выбрано новых примеров: {len(examples)}; "
            f"активных сейчас: {len(active)}; "
            f"режим: {'APPLY' if args.apply else 'PREVIEW'}."
        )
        if not args.apply:
            print("БД не изменена. Для замены добавьте --apply.")
            return

        for memory in active:
            memory.source_type = "style_example_archived"
        for example in examples:
            session.add(
                TwinMemory(
                    twin_id=args.twin_id,
                    content=example,
                    embedding=None,
                    embedding_model=None,
                    source_type="style_example",
                    source_author="unknown",
                    source_date=None,
                )
            )
        session.commit()
        print(
            f"Активные примеры заменены: добавлено={len(examples)}, "
            f"архивировано={len(active)}; другие записи не изменялись."
        )


if __name__ == "__main__":
    main()
