import json
import unittest
from unittest.mock import MagicMock, patch

from services.core.chat_service import (
    _build_context_messages,
    _format_retrieved_memories,
    _previous_crisis_score,
    RetrievedMemory,
    needs_search_query_rewrite,
    prepare_search_query,
    retrieve,
    should_search_memory,
)
from services.core.onboarding_service import (
    MAX_MEMORY_CHUNK_CHARACTERS,
    _create_profile,
    _conversation_chunks,
    _parse_conversation,
    select_style_examples,
)
from db.models import TwinMemory, TwinProfile
from services.core.prompt_builder import (
    PROFILE_DATA_END,
    PROFILE_DATA_START,
    STYLE_DATA_END,
    STYLE_DATA_START,
    build_system_prompt,
)


class SearchContextTests(unittest.TestCase):
    def test_greeting_and_unrelated_questions_skip_memory_search(self) -> None:
        self.assertFalse(should_search_memory("Привет"))
        self.assertFalse(should_search_memory("Добрый день, как дела?"))
        self.assertFalse(should_search_memory("Сколько будет 2 + 2?"))
        self.assertFalse(should_search_memory("Почему мы так страдаем?"))
        self.assertTrue(should_search_memory("Привет, помнишь поездку в Сочи?"))
        self.assertTrue(should_search_memory("А кто тогда был с нами?"))
        self.assertTrue(should_search_memory("Какая у тебя любимая музыка?"))
        self.assertTrue(should_search_memory("Что ты думаешь о фотографии?"))
        self.assertTrue(should_search_memory("Какие цветы тебе нравились?"))
        history = [{"role": "user", "content": "Мы обсуждали сад у дома."}]
        self.assertTrue(should_search_memory("А что насчёт этого?", history))

    def test_only_contextual_followups_are_rewritten(self) -> None:
        history = [
            {"role": "user", "content": "Вспоминаю поездку в Сочи."},
            {"role": "assistant", "content": "Что именно хочешь вспомнить?"},
        ]
        self.assertTrue(needs_search_query_rewrite("А кто тогда был с нами?", history))
        self.assertFalse(needs_search_query_rewrite("Кто был с нами в Сочи?", []))
        self.assertFalse(needs_search_query_rewrite("Какая его любимая книга?", []))

    def test_rewrite_uses_history_and_falls_back_to_original_when_not_needed(self) -> None:
        with patch("services.core.chat_service.get_ollama_client") as get_client:
            original = "А кто тогда был с нами?"
            self.assertEqual(prepare_search_query(original, []), original)
            get_client.assert_not_called()

            client = get_client.return_value
            client.chat.return_value = {
                "message": {"content": "Кто был с нами во время поездки в Сочи?"}
            }
            history = [{"role": "user", "content": "Расскажи о поездке в Сочи."}]
            rewritten = prepare_search_query(original, history)
            self.assertEqual(rewritten, "Кто был с нами во время поездки в Сочи?")
            self.assertEqual(original, "А кто тогда был с нами?")
            sent_history = client.chat.call_args.kwargs["messages"][1]["content"]
            self.assertIn("поездке в Сочи", sent_history)
            self.assertIn(original, sent_history)

    def test_retrieve_returns_limitation_when_no_memories_match(self) -> None:
        session_context = MagicMock()
        session = session_context.__enter__.return_value
        session.execute.side_effect = [
            MagicMock(all=lambda: []),
            MagicMock(first=lambda: None),
        ]
        with (
            patch("services.core.chat_service.get_ollama_client") as get_client,
            patch("services.core.chat_service.SessionLocal", return_value=session_context),
        ):
            get_client.return_value.embeddings.return_value = {"embedding": [0.0] * 1024}
            result = retrieve("synthetic query", -1)

        self.assertEqual(result.memories, ())
        self.assertIsNotNone(result.limitation)

    def test_crisis_score_is_scoped_to_profile_and_user(self) -> None:
        session_context = MagicMock()
        session = session_context.__enter__.return_value
        session.execute.return_value.scalar_one_or_none.return_value = 2
        with patch(
            "services.core.chat_service.SessionLocal",
            return_value=session_context,
        ):
            self.assertEqual(_previous_crisis_score(12, "tg:34"), 2)

        statement = session.execute.call_args.args[0]
        sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("chat_messages.twin_id = 12", sql)
        self.assertIn("chat_messages.user_id = 'tg:34'", sql)


class UploadedMaterialTests(unittest.TestCase):
    def test_legacy_uploaded_material_author_is_presented_as_unknown(self) -> None:
        memory = RetrievedMemory(
            content="Synthetic imported fragment.",
            similarity=0.8,
            source_type="uploaded_material",
            source_author="subject_material",
            source_date=None,
            embedding_model="qwen3-embedding:0.6b",
        )

        rendered = _format_retrieved_memories((memory,))

        self.assertIn("origin=unknown", rendered)
        self.assertNotIn("origin=subject_material", rendered)

    def test_uploaded_material_author_is_not_inferred(self) -> None:
        class InMemorySession:
            def __init__(self) -> None:
                self.added = []

            def add(self, item) -> None:
                self.added.append(item)

            def flush(self) -> None:
                next(item for item in self.added if isinstance(item, TwinProfile)).id = 1

        session = InMemorySession()
        _create_profile(
            session,
            {"subject.alias": "Synthetic profile"},
            "Unlabeled synthetic conversation paragraph.",
            lambda _: [0.0] * 1024,
        )
        memories = [
            item for item in session.added
            if isinstance(item, TwinMemory) and item.source_type == "uploaded_material"
        ]

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].source_author, "unknown")
        self.assertIsNone(memories[0].source_date)

    def test_colons_and_blank_lines_are_not_treated_as_speaker_boundaries(self) -> None:
        source = (
            "Письмо: сегодня я вспоминал поездку.\n"
            "В этом предложении есть время: 18:30.\n\n"
            "Второй абзац того же письма."
        )
        paragraphs = _parse_conversation(source)
        self.assertEqual(len(paragraphs), 2)
        self.assertIn("время: 18:30", paragraphs[0])

    def test_long_text_is_chunked_with_overlap_without_dropping_sentences(self) -> None:
        sentences = [f"Тема номер {index} содержит искусственное описание." for index in range(90)]
        paragraphs = _parse_conversation("\n\n".join(sentences))
        chunks = _conversation_chunks(paragraphs)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.content) <= MAX_MEMORY_CHUNK_CHARACTERS for chunk in chunks))
        combined = "\n".join(chunk.content for chunk in chunks)
        for sentence in sentences:
            self.assertIn(sentence, combined)
        self.assertEqual([chunk.sequence for chunk in chunks], list(range(1, len(chunks) + 1)))

    def test_style_examples_skip_quotes_and_reported_speech_and_vary(self) -> None:
        source = "\n\n".join([
            "«Я всегда говорю коротко и по делу», — написал знакомый.",
            "Он мне сказал: я бы выбрал другой маршрут и не стал торопиться.",
            "По воскресеньям люблю идти пешком через тихий парк, даже если дорога получается длиннее.",
            "А потом можно зайти за горячим чаем, посидеть у окна и посмотреть, как дождь меняет цвет улицы.",
            "Мне нравится, когда разговор не торопится и можно спокойно перескочить с одной мысли на другую.",
            "Иногда план на день держится на одной хорошей песне и чашке крепкого кофе.",
        ])
        examples = select_style_examples(source)

        self.assertGreaterEqual(len(examples), 3)
        self.assertLessEqual(len(examples), 5)
        self.assertTrue(all("«" not in item and "»" not in item for item in examples))
        self.assertTrue(all("Он мне сказал" not in item for item in examples))
        self.assertEqual(len(set(examples)), len(examples))


class ContextBudgetTests(unittest.TestCase):
    def test_required_system_and_current_message_survive_budgeting(self) -> None:
        system = (
            "Обязательные правила безопасности."
            f"\n{PROFILE_DATA_START}\n"
            + "\n".join(json.dumps({"answer": str(index)}) for index in range(200))
            + f"\n{PROFILE_DATA_END}\n"
            "Правило цифрового образа."
        )
        with patch.dict(
            "os.environ",
            {
                "OLLAMA_NUM_CTX": "1024",
                "OLLAMA_RESPONSE_RESERVE_TOKENS": "900",
                "MAX_PROFILE_CONTEXT_CHARS": "10000",
            },
        ):
            messages = _build_context_messages(
                system,
                [{"role": "user", "content": "old history"}] * 8,
                "Текущий вопрос должен сохраниться.",
                (),
                None,
            )
        self.assertIn("Обязательные правила безопасности.", messages[0]["content"])
        self.assertIn("Правило цифрового образа.", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "Текущий вопрос должен сохраниться.")

    def test_small_source_uses_full_text_only_when_total_budget_allows_it(self) -> None:
        prompt = (
            "Обязательные инструкции."
            f"\n{PROFILE_DATA_START}\n{PROFILE_DATA_END}\n"
            f"\n{STYLE_DATA_START}\n{STYLE_DATA_END}\n"
        )
        source_text = "synthetic source " * 900
        with patch.dict(
            "os.environ",
            {
                "OLLAMA_CHARS_PER_TOKEN": "1.5",
                "OLLAMA_RESPONSE_RESERVE_TOKENS": "1024",
                "MAX_PROFILE_CONTEXT_CHARS": "4000",
                "MAX_STYLE_CONTEXT_CHARS": "1600",
                "MAX_MEMORY_CONTEXT_CHARS": "3000",
            },
        ):
            with patch.dict("os.environ", {"OLLAMA_NUM_CTX": "8192"}):
                small_window = _build_context_messages(
                    prompt, [], "Где это было?", (), None, source_text
                )
            with patch.dict("os.environ", {"OLLAMA_NUM_CTX": "16384"}):
                large_window = _build_context_messages(
                    prompt, [], "Где это было?", (), None, source_text
                )

        self.assertNotIn("UNTRUSTED_FULL_SOURCE", "\n".join(
            message["content"] for message in small_window
        ))
        self.assertIn("UNTRUSTED_FULL_SOURCE", "\n".join(
            message["content"] for message in large_window
        ))
        self.assertEqual(large_window[-1]["content"], "Где это было?")

    def test_style_samples_are_a_separate_non_factual_context_block(self) -> None:
        prompt = (
            "System rules."
            f"\n{PROFILE_DATA_START}\n{PROFILE_DATA_END}\n"
            f"\n{STYLE_DATA_START}\n"
            '{"category":"STYLE_EXAMPLE_CANDIDATE","text":"Synthetic style sample."}\n'
            f"{STYLE_DATA_END}\n"
        )
        messages = _build_context_messages(
            prompt,
            [],
            "Current question.",
            (),
            None,
        )

        self.assertFalse(any(STYLE_DATA_START in item["content"] for item in messages))
        self.assertTrue(any("Synthetic style sample." in item["content"] for item in messages))
        self.assertTrue(any("не биографические факты" in item["content"] for item in messages))


class PromptVoiceTests(unittest.TestCase):
    def test_identity_disclosure_is_reserved_for_direct_questions(self) -> None:
        session_context = MagicMock()
        session = session_context.__enter__.return_value
        session.get.return_value.name = "Synthetic twin"
        session.execute.return_value.scalars.return_value.all.return_value = []
        with patch(
            "services.core.prompt_builder.SessionLocal",
            return_value=session_context,
        ):
            prompt = build_system_prompt(1)

        self.assertIn("не начинай ответы с шаблонного объяснения", prompt)
        self.assertIn("если собеседник прямо спрашивает, ты ли это", prompt)
        self.assertIn("не добавляй без прямого вопроса оговорку о невозможности быть рядом", prompt)
        self.assertNotIn("Ты не сам этот человек", prompt)
        self.assertNotIn("У тебя нет собственного сознания", prompt)


if __name__ == "__main__":
    unittest.main()
