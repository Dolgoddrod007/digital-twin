import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from db.connection import SessionLocal
from db.models import ClientOnboarding, TwinAttribute, TwinMemory, TwinProfile
from services.core.ollama_client import get_ollama_client

logger = logging.getLogger(__name__)
CONSENT_VERSION = "messenger-onboarding-2"
MAX_CONVERSATION_FILE_BYTES = 20 * 1024 * 1024
MAX_CONVERSATION_CHARACTERS = 200_000
MAX_MEMORY_CHUNK_CHARACTERS = 800
_QUESTIONNAIRE_PATH = Path(__file__).resolve().parents[2] / "config" / "questionnaire.json"
_QUESTIONNAIRE = json.loads(_QUESTIONNAIRE_PATH.read_text(encoding="utf-8"))
QUESTIONS = _QUESTIONNAIRE["questions"]
QUESTION_BLOCKS = (
    (0, 1, "Обращение"),
    (1, 12, "Манера общения и стиль речи"),
    (12, 20, "Характер и реакции"),
    (20, 25, "Окружение и картина мира"),
    (25, 27, "Фундаментальные вопросы"),
)
EmbeddingFactory = Callable[[str], list[float]]

GENERAL_CONSENT_TEXT = (
    "Перед началом: вы общаетесь с программной языковой моделью, созданной по профилю, "
    "а не с умершим человеком. Ответы будут отправляться через инфраструктуру VK или "
    "Telegram и сохраняться в базе проекта. Сейчас сообщения и профиль хранятся без "
    "шифрования. Не указывайте сведения о здоровье, обстоятельствах смерти, кризисах "
    "или личные данные других людей. Вы можете пропустить необязательный вопрос "
    "или остановить заполнение командой /pause. Согласны продолжить?"
)
@dataclass(frozen=True)
class OnboardingReply:
    text: str
    buttons: tuple[tuple[str, str], ...] = ()


def _action(value: str) -> str:
    action = value.strip().lower()
    if action.startswith("/"):
        action = action.split(maxsplit=1)[0].split("@", maxsplit=1)[0].removeprefix("/")
    return {
        "согласен, продолжить": "consent_yes",
        "не согласен": "consent_no",
        "docx": "unsupported_file",
        "добавить переписку из txt": "chat_file_yes",
        "продолжить без файла": "chat_file_no",
        "назад": "back",
        "пропустить": "skip",
        "сохранить и выйти": "pause",
        "отменить загрузку": "cancel_upload",
        "отменить заполнение": "cancel",
    }.get(action, action)


def _load_answers(session: ClientOnboarding) -> dict[str, str]:
    return json.loads(session.answers_json or "{}")


def _store_answers(session: ClientOnboarding, answers: dict[str, str]) -> None:
    session.answers_json = json.dumps(answers, ensure_ascii=False)


def _question_block(step_index: int) -> tuple[int, str]:
    for block_number, (start, end, title) in enumerate(QUESTION_BLOCKS, start=1):
        if start <= step_index < end:
            return block_number, title
    raise IndexError(f"Question index {step_index} is outside the questionnaire")


def _question_reply(session: ClientOnboarding) -> OnboardingReply:
    question = QUESTIONS[session.step_index]
    block_number, block_title = _question_block(session.step_index)
    buttons: list[tuple[str, str]] = []
    if not question["required"]:
        buttons.append(("Пропустить", "skip"))
    if session.step_index > 0:
        buttons.append(("Назад", "back"))
    buttons.extend((("Сохранить и выйти", "pause"), ("Отменить заполнение", "cancel")))
    return OnboardingReply(
        text=(
            f"Блок {block_number}. {block_title}\n"
            f"Вопрос {session.step_index + 1} из {len(QUESTIONS)}\n\n"
            f"{question['prompt']}\n\n"
            "Ответьте текстом или воспользуйтесь кнопками."
        ),
        buttons=tuple(buttons),
    )


def _needs_legacy_alias(
    session: ClientOnboarding,
    answers: dict[str, str],
) -> bool:
    return (
        session.twin_id is None
        and session.consent_signed_at is not None
        and "subject.alias" not in answers
        and any(question["id"] in answers for question in QUESTIONS[1:])
    )


def _legacy_alias_reply() -> OnboardingReply:
    return OnboardingReply(
        "Перед продолжением анкеты уточним, как называть человека в профиле. "
        f"{QUESTIONS[0]['prompt']}\n\nОтветьте текстом.",
        (("Сохранить и выйти", "pause"), ("Отменить заполнение", "cancel")),
    )


def _request_legacy_alias(db, session: ClientOnboarding) -> OnboardingReply:
    session.previous_status = session.status
    session.status = "awaiting_subject_alias"
    db.commit()
    return _legacy_alias_reply()


def _complete_legacy_alias(
    db,
    session: ClientOnboarding,
    answer: str,
) -> OnboardingReply:
    answers = _load_answers(session)
    answers["subject.alias"] = answer
    _store_answers(session, answers)
    previous_status = session.previous_status
    session.previous_status = None
    session.step_index = next(
        (
            index
            for index, question in enumerate(QUESTIONS)
            if question["id"] not in answers
        ),
        len(QUESTIONS),
    )

    if session.step_index < len(QUESTIONS):
        session.status = "form"
        db.commit()
        return _question_reply(session)

    if previous_status == "awaiting_chat_file":
        session.status = "awaiting_chat_file"
        db.commit()
        return _chat_file_upload_reply()
    if previous_status == "awaiting_chat_file_choice":
        session.status = "awaiting_chat_file_choice"
        db.commit()
        return _chat_file_choice_reply()
    if previous_status in ("awaiting_profile_creation", "awaiting_profile_confirmation"):
        session.status = "awaiting_profile_creation"
        db.commit()
        return _activate_profile(db, session.user_id, session)

    session.status = "awaiting_chat_file_choice"
    db.commit()
    return _chat_file_choice_reply()


def _consent_reply() -> OnboardingReply:
    return OnboardingReply(
        GENERAL_CONSENT_TEXT,
        (("Согласен, продолжить", "consent_yes"), ("Не согласен", "consent_no")),
    )


def _stage_reply(session: ClientOnboarding) -> OnboardingReply:
    if session.status == "awaiting_subject_alias":
        return _legacy_alias_reply()
    if session.status == "awaiting_chat_file_choice":
        return _chat_file_choice_reply()
    if session.status == "awaiting_chat_file":
        return _chat_file_upload_reply()
    if session.status == "awaiting_profile_creation":
        return OnboardingReply(
            "Ответы анкеты сохранены, но профиль не удалось создать. "
            "Отправьте /continue, чтобы повторить создание."
        )
    return _question_reply(session)


def _chat_file_choice_reply() -> OnboardingReply:
    return OnboardingReply(
        "Анкета заполнена. Если хотите, добавьте переписку в виде TXT-файла. "
        "Файл будет передан через мессенджер и сохранён в профиле без шифрования; "
        "перед отправкой удалите личные данные и сведения, которые не хотите сохранять, "
        "и загружайте только переписку, которую имеете право передавать. "
        "Можно продолжить без файла.",
        (
            ("Добавить переписку из TXT", "chat_file_yes"),
            ("Продолжить без файла", "chat_file_no"),
        ),
    )


def _chat_file_upload_reply() -> OnboardingReply:
    return OnboardingReply(
        f"Пришлите TXT-файл с перепиской. Максимальный размер — "
        f"{MAX_CONVERSATION_FILE_BYTES // (1024 * 1024)} МБ и "
        f"{MAX_CONVERSATION_CHARACTERS} символов. "
        "После получения сообщу, когда обработка завершится. "
        "Чтобы продолжить без файла, отправьте /cancel_upload.",
        (("Продолжить без файла", "cancel_upload"),),
    )


def _get_or_create_session(db, user_id: str, channel: str) -> ClientOnboarding:
    onboarding = db.get(ClientOnboarding, user_id)
    if onboarding is None:
        onboarding = ClientOnboarding(
            user_id=user_id,
            channel=channel,
            status="new",
            step_index=0,
            answers_json="{}",
        )
        db.add(onboarding)
        db.flush()
    return onboarding


def handle_message(user_id: str, channel: str, text: str) -> OnboardingReply | None:
    """Handle onboarding commands and answers; return None only when chat is active."""
    action = _action(text)
    with SessionLocal() as db:
        onboarding = _get_or_create_session(db, user_id, channel)
        if (
            onboarding.twin_id is None
            and onboarding.consent_signed_at is not None
            and onboarding.consent_version != CONSENT_VERSION
        ):
            onboarding.status = "new"
            onboarding.step_index = 0
            onboarding.conversation_text = None
            onboarding.conversation_approved = 0
            onboarding.consent_version = None
            onboarding.consent_signed_at = None
            onboarding.previous_status = None
            _store_answers(onboarding, {})
            db.commit()
        if onboarding.twin_id is None:
            answers = _load_answers(onboarding)
            if "style.greeting" in answers:
                answers.pop("style.greeting")
                if onboarding.step_index > 11:
                    onboarding.step_index -= 1
                _store_answers(onboarding, answers)
                if onboarding.status == "awaiting_profile_confirmation":
                    onboarding.status = "awaiting_chat_file_choice"
                    onboarding.conversation_text = None
                    onboarding.conversation_approved = 0
                db.commit()

        if action == "unsupported_file":
            return OnboardingReply(
                "Файлы можно загрузить только в формате TXT и только после заполнения анкеты."
            )

        if action == "new":
            if onboarding.status == "active" and onboarding.twin_id is not None:
                onboarding.twin_id = None
                onboarding.status = "awaiting_consent"
                onboarding.step_index = 0
                _store_answers(onboarding, {})
                onboarding.conversation_text = None
                onboarding.conversation_approved = 0
                onboarding.previous_status = None
                onboarding.consent_version = None
                onboarding.consent_signed_at = None
                db.commit()
                consent_reply = _consent_reply()
                return OnboardingReply(
                    "Начинаем создание нового двойника. \n\n"
                    + consent_reply.text,
                    consent_reply.buttons,
                )
            if onboarding.status in ("new", "cancelled", "awaiting_consent"):
                onboarding.status = "awaiting_consent"
                db.commit()
                return _consent_reply()
            return OnboardingReply(
                "Сначала завершите текущую анкету командой /continue или отмените её командой /cancel."
            )

        if onboarding.status == "awaiting_subject_alias":
            if action in {
                "start",
                "begin",
                "continue",
                "back",
                "skip",
                "consent_yes",
                "consent_no",
                "chat_file_yes",
                "chat_file_no",
                "cancel_upload",
            }:
                return _legacy_alias_reply()
            if action not in {"help", "status", "cancel", "pause"}:
                answer = text.strip()
                question = QUESTIONS[0]
                if not answer:
                    return OnboardingReply(
                        "Нужен непустой ответ. Можно указать имя, псевдоним или оба варианта."
                    )
                if len(answer) > question["max_length"]:
                    return OnboardingReply(
                        f"Ответ слишком длинный. Максимум — {question['max_length']} символов."
                    )
                return _complete_legacy_alias(db, onboarding, answer)

        if action in ("start", "begin"):
            if onboarding.twin_id is not None:
                return OnboardingReply("Профиль готов. Напишите сообщение, чтобы начать диалог.")
            if _needs_legacy_alias(onboarding, _load_answers(onboarding)):
                return _request_legacy_alias(db, onboarding)
            if onboarding.status in ("awaiting_profile_confirmation", "awaiting_profile_creation"):
                onboarding.status = "awaiting_profile_creation"
                db.commit()
                return _activate_profile(db, user_id, onboarding)
            if onboarding.consent_signed_at is None:
                onboarding.status = "awaiting_consent"
                db.commit()
                return _consent_reply()
            if onboarding.status == "paused":
                return OnboardingReply("Черновик сохранён. Чтобы продолжить, отправьте /continue.")
            return _stage_reply(onboarding)

        if action == "help":
            return OnboardingReply(
                "Команды: /start — начать; /continue — продолжить; /pause — сохранить черновик; "
                "/cancel — удалить незавершённый черновик; /new — создать новый профиль."
            )

        if action == "status":
            if onboarding.twin_id is not None:
                return OnboardingReply("Профиль готов, диалог доступен.")
            return OnboardingReply(f"Состояние заполнения: {onboarding.status}.")

        if action == "cancel":
            if onboarding.twin_id is not None:
                return OnboardingReply("Профиль уже создан; отмена заполнения к нему неприменима.")
            onboarding.status = "cancelled"
            onboarding.step_index = 0
            onboarding.conversation_text = None
            onboarding.conversation_approved = 0
            _store_answers(onboarding, {})
            db.commit()
            return OnboardingReply("Незавершённый черновик удалён. Для начала отправьте /start.")

        if (
            onboarding.status != "awaiting_subject_alias"
            and action not in {"pause", "cancel_upload"}
            and _needs_legacy_alias(onboarding, _load_answers(onboarding))
        ):
            return _request_legacy_alias(db, onboarding)

        if onboarding.status == "awaiting_profile_confirmation":
            onboarding.status = "awaiting_chat_file_choice"
            onboarding.conversation_text = None
            onboarding.conversation_approved = 0
            db.commit()
            return _chat_file_choice_reply()

        if onboarding.status == "awaiting_profile_creation":
            if action == "continue":
                return _activate_profile(db, user_id, onboarding)
            return _stage_reply(onboarding)

        if action == "pause":
            if onboarding.twin_id is not None:
                return OnboardingReply("Профиль уже готов. Чтобы приостановить диалог, просто не отправляйте сообщения.")
            if onboarding.consent_signed_at is None:
                return OnboardingReply("Заполнение ещё не начато. Отправьте /start, чтобы прочитать условия.")
            onboarding.status = "paused"
            db.commit()
            return OnboardingReply("Черновик сохранён. Вернуться можно командой /continue.")

        if action == "continue":
            if onboarding.twin_id is not None:
                return OnboardingReply("Профиль готов. Напишите сообщение, чтобы продолжить диалог.")
            if onboarding.consent_signed_at is None:
                return OnboardingReply("Сначала отправьте /start и подтвердите условия.")
            if _needs_legacy_alias(onboarding, _load_answers(onboarding)):
                return _request_legacy_alias(db, onboarding)
            if onboarding.status == "paused":
                onboarding.status = "form"
                db.commit()
            return _stage_reply(onboarding)

        if onboarding.status == "new" or onboarding.status == "cancelled":
            onboarding.status = "awaiting_consent"
            db.commit()
            return _consent_reply()

        if onboarding.status == "awaiting_consent":
            if action == "consent_yes":
                onboarding.consent_version = CONSENT_VERSION
                onboarding.consent_signed_at = datetime.now(timezone.utc)
                onboarding.status = "form"
                db.commit()
                return _question_reply(onboarding)
            if action in ("consent_no", "cancel"):
                onboarding.status = "cancelled"
                db.commit()
                return OnboardingReply("Без согласия анкета и диалог недоступны. Данные не собираются.")
            return OnboardingReply(
                "Чтобы продолжить, выберите один из вариантов ниже.",
                (("Согласен, продолжить", "consent_yes"), ("Не согласен", "consent_no")),
            )

        if onboarding.status == "awaiting_chat_file_choice":
            if action == "chat_file_yes":
                onboarding.status = "awaiting_chat_file"
                db.commit()
                return _chat_file_upload_reply()
            if action == "chat_file_no":
                onboarding.status = "awaiting_profile_creation"
                db.commit()
                return _activate_profile(db, user_id, onboarding)
            return _chat_file_choice_reply()

        if action == "cancel_upload" and onboarding.status == "awaiting_chat_file":
            onboarding.conversation_text = None
            onboarding.conversation_approved = 0
            onboarding.status = "awaiting_profile_creation"
            db.commit()
            return _activate_profile(db, user_id, onboarding)

        if onboarding.status == "awaiting_chat_file":
            return _chat_file_upload_reply()

        if onboarding.status == "active":
            if action in {
                "consent_yes",
                "consent_no",
                "chat_file_yes",
                "chat_file_no",
                "back",
                "skip",
                "pause",
                "cancel",
            }:
                return OnboardingReply("Профиль уже готов. Напишите сообщение, чтобы начать диалог.")
            return None

        if onboarding.twin_id is not None:
            return None

        if onboarding.status == "paused":
            return OnboardingReply("Черновик сохранён. Для продолжения отправьте /continue.")

        if action == "back":
            if onboarding.step_index <= 0:
                return _question_reply(onboarding)
            onboarding.step_index -= 1
            onboarding.status = "form"
            db.commit()
            return _question_reply(onboarding)

        if onboarding.status != "form":
            onboarding.status = "form"

        if onboarding.step_index >= len(QUESTIONS):
            onboarding.status = "awaiting_chat_file_choice"
            db.commit()
            return _chat_file_choice_reply()

        if action in {
            "consent_yes",
            "consent_no",
            "chat_file_yes",
            "chat_file_no",
        }:
            return _question_reply(onboarding)

        question = QUESTIONS[onboarding.step_index]
        if action == "skip":
            if question["required"]:
                return OnboardingReply("Этот вопрос обязательный. Можно ответить коротко или отправить /pause.")
            answer = ""
        else:
            answer = text.strip()

        if question["required"] and not answer:
            return OnboardingReply("Нужен непустой ответ. Или остановите заполнение командой /pause.")
        if len(answer) > question["max_length"]:
            return OnboardingReply(
                f"Ответ слишком длинный. Максимум — {question['max_length']} символов."
            )

        answers = _load_answers(onboarding)
        answers[question["id"]] = answer
        _store_answers(onboarding, answers)
        onboarding.step_index += 1

        if onboarding.step_index < len(QUESTIONS):
            db.commit()
            return _question_reply(onboarding)

        onboarding.status = "awaiting_chat_file_choice"
        db.commit()
        return _chat_file_choice_reply()


def _activate_profile(db, user_id: str, onboarding: ClientOnboarding) -> OnboardingReply:
    if onboarding.twin_id is not None:
        return OnboardingReply("Профиль уже готов. Напишите сообщение, чтобы начать диалог.")
    if _needs_legacy_alias(onboarding, _load_answers(onboarding)):
        return _request_legacy_alias(db, onboarding)
    if onboarding.step_index < len(QUESTIONS):
        return _question_reply(onboarding)

    answers = _load_answers(onboarding)
    onboarding.status = "awaiting_profile_creation"
    db.commit()
    try:
        twin_id, embeddings_unavailable = _create_profile(
            db,
            answers,
            onboarding.conversation_text if onboarding.conversation_approved else None,
            embedding_factory=_embedding,
        )
    except Exception as error:
        logger.error("Profile creation failed (error_type=%s)", type(error).__name__)
        db.rollback()
        onboarding = db.get(ClientOnboarding, user_id)
        if onboarding is None:
            raise RuntimeError("Onboarding draft disappeared during profile creation") from error
        onboarding.status = "awaiting_profile_creation"
        db.commit()
        logger.error(
            "Profile creation failed (error_type=%s)",
            type(error).__name__,
        )
        return OnboardingReply(
            "Ответы анкеты сохранены, но не удалось создать профиль. "
            "Проверьте подключение к PostgreSQL и Ollama, затем отправьте /continue."
        )

    onboarding.twin_id = twin_id
    onboarding.status = "active"
    onboarding.conversation_text = None
    onboarding.conversation_approved = 0
    db.commit()
    if embeddings_unavailable:
        return OnboardingReply(
            "Анкета сохранена, профиль создан, можно начинать диалог. "
            "Сервис эмбеддингов Ollama сейчас недоступен, поэтому поиск по воспоминаниям "
            "может быть ограничен."
        )
    return OnboardingReply(
        "Профиль создан. Можно начинать диалог — напишите первое сообщение. "
        "Помните: это языковая модель по профилю, а не сам человек."
    )


def get_twin_id(user_id: str) -> int | None:
    with SessionLocal() as db:
        onboarding = db.get(ClientOnboarding, user_id)
        if onboarding is None or onboarding.status != "active":
            return None
        return onboarding.twin_id


def _embedding(text: str) -> list[float]:
    return get_ollama_client().embeddings(
        model=os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b"),
        prompt=text,
    )["embedding"]


def _memory_chunks(text: str) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > MAX_MEMORY_CHUNK_CHARACTERS:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:MAX_MEMORY_CHUNK_CHARACTERS])
            paragraph = paragraph[MAX_MEMORY_CHUNK_CHARACTERS:]
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 1 > MAX_MEMORY_CHUNK_CHARACTERS:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def _create_profile(
    db,
    answers: dict[str, str],
    conversation_text: str | None,
    embedding_factory: EmbeddingFactory,
) -> tuple[int, bool]:
    alias = answers.get("subject.alias", "Цифровой двойник").strip() or "Цифровой двойник"
    profile = TwinProfile(
        name=alias,
        status="active",
        current_state="ACTIVE",
        activated_at=datetime.now(timezone.utc),
    )
    db.add(profile)
    db.flush()

    for key, value in answers.items():
        if key == "deep.memory_triggers" or not value:
            continue
        block, attribute_key = key.split(".", 1)
        db.add(
            TwinAttribute(
                twin_id=profile.id,
                block=block,
                attribute_key=attribute_key,
                raw_answer=value,
                prompt_fragment=f"{attribute_key}: {value}",
            )
        )

    memories = answers.get("deep.memory_triggers", "")
    if conversation_text:
        memories = f"{memories}\n{conversation_text}".strip()
    embeddings_unavailable = False
    for chunk in _memory_chunks(memories):
        vector = None
        if not embeddings_unavailable:
            try:
                vector = embedding_factory(chunk)
            except Exception as error:
                embeddings_unavailable = True
                logger.error(
                    "Memory embedding failed (error_type=%s); profile will be saved without embeddings",
                    type(error).__name__,
                )
        db.add(TwinMemory(twin_id=profile.id, content=chunk, embedding=vector))

    return profile.id, embeddings_unavailable


def load_conversation_text(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() != ".txt":
        raise ValueError("Поддерживаются только файлы .txt")
    if file_path.stat().st_size > MAX_CONVERSATION_FILE_BYTES:
        raise ValueError(
            f"Файл превышает лимит {MAX_CONVERSATION_FILE_BYTES // (1024 * 1024)} МБ"
        )

    raw_text = file_path.read_bytes()
    if b"\x00" in raw_text:
        raise ValueError("Файл должен содержать обычный текст в формате .txt")
    try:
        text = raw_text.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw_text.decode("cp1251")
        except UnicodeDecodeError:
            raise ValueError("Не удалось прочитать TXT-файл. Сохраните его в UTF-8.") from None
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("В TXT-файле не найден текст")
    if len(text) > MAX_CONVERSATION_CHARACTERS:
        raise ValueError(
            f"В файле больше {MAX_CONVERSATION_CHARACTERS} символов; сократите переписку"
        )
    return text


def save_conversation_draft(user_id: str, text: str) -> OnboardingReply:
    if len(text) > MAX_CONVERSATION_CHARACTERS:
        raise ValueError(f"В файле больше {MAX_CONVERSATION_CHARACTERS} символов")
    with SessionLocal() as db:
        onboarding = db.get(ClientOnboarding, user_id)
        if onboarding is None or onboarding.status != "awaiting_chat_file":
            raise ValueError("Сначала завершите анкету и выберите загрузку переписки TXT")
        onboarding.conversation_text = text
        onboarding.conversation_approved = 1
        onboarding.status = "awaiting_profile_creation"
        db.commit()
        activated = _activate_profile(db, user_id, onboarding)
    paragraph_count = sum(1 for line in text.splitlines() if line.strip())
    return OnboardingReply(
        f"Обработка TXT завершена: {len(text)} символов, непустых строк: "
        f"{paragraph_count}.\n\n{activated.text}",
        activated.buttons,
    )
