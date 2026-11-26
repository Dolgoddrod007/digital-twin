import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from db.connection import SessionLocal
from db.models import TwinAttribute, TwinMemory, TwinProfile
from services.core.chat_service import (
    _build_context_messages,
    _load_chat_history,
    _load_source_text,
    prepare_search_query,
    retrieve,
    should_search_memory,
    _embedding_model,
)
from services.core.prompt_builder import (
    build_system_prompt,
)


def _included_profile_keys(messages: list[dict[str, str]]) -> set[str]:
    keys: set[str] = set()
    for message in messages:
        if message["role"] != "system":
            continue
        if "Следующие строки — данные анкеты" not in message["content"]:
            continue
        for line in message["content"].splitlines()[1:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            field_id = item.get("field_id")
            if field_id:
                keys.add(str(field_id))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Локальная диагностика контекста без вывода текстов анкеты и переписки"
    )
    parser.add_argument("--twin-id", required=True, type=int)
    parser.add_argument(
        "--query",
        required=True,
        help="Вопрос для локального предварительного просмотра; содержимое не печатается",
    )
    parser.add_argument(
        "--user-id",
        help="ID диалога (например, tg:123); без него история не включается в предварительный просмотр",
    )
    args = parser.parse_args()

    with SessionLocal() as session:
        profile_exists = session.get(TwinProfile, args.twin_id) is not None
        stored_keys = session.execute(
            select(TwinAttribute.block, TwinAttribute.attribute_key)
            .where(TwinAttribute.twin_id == args.twin_id, TwinAttribute.raw_answer.is_not(None), TwinAttribute.raw_answer != "")
        ).all()
        memory_counts = session.execute(
            select(TwinMemory.source_type, func.count(TwinMemory.id))
            .where(TwinMemory.twin_id == args.twin_id)
            .group_by(TwinMemory.source_type)
        ).all()
        compatible_count = session.execute(
            select(func.count(TwinMemory.id)).where(
                TwinMemory.twin_id == args.twin_id,
                TwinMemory.embedding.is_not(None),
                TwinMemory.embedding_model == _embedding_model(),
                TwinMemory.source_type.not_in(("style_example", "style_example_archived"))
                | TwinMemory.source_type.is_(None),
            )
        ).scalar_one()

    print(f"profile_exists={profile_exists}")
    if not profile_exists:
        return

    keys = {f"{block}.{key}" for block, key in stored_keys}
    keys.add("subject.alias")
    print(f"questionnaire_fields_saved={len(keys)}")
    print(f"source_text_saved={bool(_load_source_text(args.twin_id))}")
    print("stored_by_type=" + (", ".join(f"{kind or 'unknown'}:{count}" for kind, count in memory_counts) or "none"))
    print(f"compatible_indexed_fragments={compatible_count}")

    history = (
        _load_chat_history(args.twin_id, args.user_id)
        if args.user_id
        else []
    )
    search_started = should_search_memory(args.query, history)
    search_query = None
    if search_started:
        search_query = prepare_search_query(args.query, history)
        retrieval = retrieve(search_query, args.twin_id)
        source_text = _load_source_text(args.twin_id)
    else:
        retrieval = None
        source_text = None

    system = build_system_prompt(args.twin_id)
    messages = _build_context_messages(
        system,
        history,
        args.query,
        retrieval.memories if retrieval else (),
        retrieval.limitation if retrieval else None,
        source_text,
        search_query=search_query,
    )
    included = _included_profile_keys(messages)

    print("profile_keys_in_prompt=" + (", ".join(sorted(included)) or "(none)"))
    print("profile_keys_omitted=" + (", ".join(sorted(keys - included)) or "(none)"))
    if not search_started:
        print("memory_search=skipped")
    else:
        if retrieval is None:
            raise RuntimeError("Memory search was selected but returned no result")
        if retrieval.limitation:
            search_status = "limited"
        else:
            search_status = "completed"
        source_types = sorted({
            memory.source_type or "unknown"
            for memory in retrieval.memories
        })
        print(f"memory_search={search_status}")
        print(f"memories_above_threshold={len(retrieval.memories)}")
        print("memory_source_types=" + (", ".join(source_types) or "(none)"))
    print(
        "history_included=%s; source_text_included=%s"
        % (
            sum(1 for message in messages if message["role"] in {"user", "assistant"}),
            any("UNTRUSTED_FULL_SOURCE" in message["content"] for message in messages),
        )
    )
    print(
        "Only counts, questionnaire keys and source types are shown; "
        "user-provided text was not printed."
    )


if __name__ == "__main__":
    main()
