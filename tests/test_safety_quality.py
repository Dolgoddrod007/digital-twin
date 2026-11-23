import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from services.core.grief_analyzer import (
    _reference_embeddings,
    analyze_message,
    classify_safety_level,
)

SCENARIOS_PATH = Path(__file__).parent / "data" / "crisis_scenarios.json"
REQUIRED_GROUPS = {
    "neutral_childhood",
    "past_hard_event",
    "grief",
    "ambiguous",
    "current_intent",
    "immediate_danger",
}


class SafetyScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_synthetic_scenarios_cover_false_positives_and_danger_misses(self) -> None:
        group_counts = Counter(example["group"] for example in self.scenarios)
        self.assertTrue(REQUIRED_GROUPS.issubset(group_counts))
        self.assertTrue(all(group_counts[group] >= 2 for group in REQUIRED_GROUPS))

        for example in self.scenarios:
            with self.subTest(scenario=example["id"]):
                self.assertEqual(
                    classify_safety_level(example["text"]),
                    example["expected"],
                )

    def test_recent_past_context_does_not_hide_explicit_current_intent(self) -> None:
        history = [
            {"role": "user", "content": "Рассказываю о тяжёлом времени в детстве."}
        ]
        self.assertEqual(classify_safety_level("Тогда я не видел смысла жить.", history), "none")
        self.assertEqual(
            classify_safety_level("Сейчас я собираюсь причинить себе вред.", history),
            "current",
        )

    def test_embedding_failure_does_not_overrule_local_safety_signal(self) -> None:
        _reference_embeddings.cache_clear()
        with patch(
            "services.core.grief_analyzer.get_ollama_client",
            side_effect=ConnectionError("synthetic outage"),
        ):
            result = analyze_message("Сейчас я собираюсь причинить себе вред.")

        self.assertEqual(result.safety_level, "current")
        self.assertTrue(result.crisis_signal)
        self.assertIsNone(result.orientation)
        self.assertFalse(result.orientation_available)

    def test_orientation_references_contain_no_crisis_class(self) -> None:
        _reference_embeddings.cache_clear()
        with patch("services.core.grief_analyzer.get_ollama_client") as get_client:
            get_client.return_value.embeddings.side_effect = [
                {"embedding": [1.0, 0.0]},
                {"embedding": [1.0, 0.0]},
                {"embedding": [0.0, 1.0]},
            ]
            result = analyze_message("Я вспоминаю спокойную прогулку.")

        requested_texts = [
            call.kwargs["prompt"]
            for call in get_client.return_value.embeddings.call_args_list
        ]
        self.assertEqual(len(requested_texts), 3)
        self.assertEqual(result.orientation, "LO")
        self.assertFalse(result.crisis_signal)
        _reference_embeddings.cache_clear()


if __name__ == "__main__":
    unittest.main()
