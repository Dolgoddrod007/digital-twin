import argparse
import json
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
        description=(
            "Оценивает порог поиска на обезличенном JSON-наборе; "
            "не выводит запросы и содержимое воспоминаний."
        )
    )
    parser.add_argument("--twin-id", required=True, type=int)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--thresholds",
        default="0.20,0.30,0.35,0.40,0.50",
        help="Список cosine similarity thresholds, разделённый запятыми",
    )
    args = parser.parse_args()

    examples = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("Ожидается непустой JSON-массив размеченных примеров")
    thresholds = [float(item.strip()) for item in args.thresholds.split(",")]
    if any(not -1.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Порог similarity должен быть в диапазоне от -1 до 1")

    model = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
    client = get_ollama_client()
    scored_examples: list[tuple[set[int], list[tuple[int, float]]]] = []
    with SessionLocal() as session:
        for example in examples:
            if (
                not isinstance(example, dict)
                or not isinstance(example.get("query"), str)
                or not isinstance(example.get("relevant_memory_ids"), list)
            ):
                raise ValueError(
                    "Каждый пример должен содержать query и relevant_memory_ids"
                )
            query_vector = client.embeddings(
                model=model,
                prompt=example["query"],
            )["embedding"]
            rows = session.execute(
                select(
                    TwinMemory.id,
                    TwinMemory.embedding.cosine_distance(query_vector).label("distance"),
                )
                .where(
                    TwinMemory.twin_id == args.twin_id,
                    TwinMemory.embedding.is_not(None),
                    TwinMemory.embedding_model == model,
                )
                .order_by(TwinMemory.embedding.cosine_distance(query_vector))
                .limit(max(1, args.top_k))
            ).all()
            scored_examples.append((
                {int(item) for item in example["relevant_memory_ids"]},
                [(int(row.id), 1.0 - float(row.distance)) for row in rows],
            ))

    print(
        f"Примеров: {len(scored_examples)}; модель: {model}; top_k={max(1, args.top_k)}"
    )
    for threshold in thresholds:
        true_positive = false_positive = false_negative = 0
        for expected, candidates in scored_examples:
            retrieved = {
                memory_id
                for memory_id, similarity in candidates
                if similarity >= threshold
            }
            true_positive += len(retrieved & expected)
            false_positive += len(retrieved - expected)
            false_negative += len(expected - retrieved)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        print(
            f"threshold={threshold:.3f}; precision={precision:.3f}; "
            f"recall={recall:.3f}; tp={true_positive}; "
            f"fp={false_positive}; fn={false_negative}"
        )


if __name__ == "__main__":
    main()
