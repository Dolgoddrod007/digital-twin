import asyncio
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import tempfile

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import CallbackQuery, ErrorEvent, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv
from services.core.onboarding_service import (
    MAX_CONVERSATION_FILE_BYTES,
    OnboardingReply,
    get_twin_id,
    handle_message as handle_onboarding_message,
    load_conversation_text,
    save_conversation_draft,
)

MAX_TELEGRAM_MESSAGE_LENGTH = 4096
logger = logging.getLogger(__name__)
ChatHandler = Callable[[str, int, str], str]


@dataclass(frozen=True)
class TelegramSettings:
    token: str


def load_settings(environ: Mapping[str, str] | None = None) -> TelegramSettings:
    env = os.environ if environ is None else environ
    token = env.get("TG_TOKEN", "").strip()
    if not token:
        raise ValueError("TG_TOKEN is required")

    return TelegramSettings(token=token)


def split_message(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> list[str]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    chunks: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit + 1)
        if split_at <= 0:
            split_at = text.rfind(" ", 0, limit + 1)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:]
    if text:
        chunks.append(text)
    return chunks


class TelegramAdapter:
    def __init__(
        self,
        settings: TelegramSettings,
        chat_handler: ChatHandler | None = None,
        onboarding_handler=handle_onboarding_message,
        twin_lookup=get_twin_id,
    ) -> None:
        self.settings = settings
        self.chat_handler = chat_handler
        self.onboarding_handler = onboarding_handler
        self.twin_lookup = twin_lookup

    async def handle_text_message(self, message: Message) -> None:
        if not message.text or message.from_user is None:
            return

        user_id = f"tg:{message.from_user.id}"
        try:
            onboarding_reply = await asyncio.to_thread(
                self.onboarding_handler,
                user_id,
                "telegram",
                message.text,
            )
        except Exception as error:
            logger.error("Telegram onboarding failed (error_type=%s)", type(error).__name__)
            await message.answer("Не удалось обработать сообщение из-за технической ошибки. Попробуйте позже.")
            return
        if onboarding_reply is not None:
            await self.send_reply(message, onboarding_reply)
            return

        try:
            twin_id = await asyncio.to_thread(self.twin_lookup, user_id)
        except Exception as error:
            logger.error("Telegram profile lookup failed (error_type=%s)", type(error).__name__)
            await message.answer("Не удалось проверить профиль из-за технической ошибки. Попробуйте позже.")
            return
        if twin_id is None:
            await message.answer("Профиль не найден. Чтобы начать заполнение, отправьте /start.")
            return

        chat_handler = self.chat_handler
        if chat_handler is None:
            from services.core.chat_service import chat

            chat_handler = chat

        processing_message = await message.answer("Печатает…")
        try:
            answer = await asyncio.to_thread(
                chat_handler,
                message.text,
                twin_id,
                user_id,
            )
        except Exception as error:
            logger.error("Telegram chat failed (error_type=%s)", type(error).__name__)
            await processing_message.edit_text(
                "Не удалось обработать сообщение из-за технической ошибки. Попробуйте позже."
            )
            return

        if not answer:
            await processing_message.delete()
            return

        chunks = split_message(answer)
        await processing_message.edit_text(chunks[0])
        for chunk in chunks[1:]:
            await message.answer(chunk)

    async def send_reply(self, message: Message, reply: OnboardingReply) -> None:
        markup = None
        if reply.buttons:
            markup = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text=label,
                            callback_data=f"onb:{action}",
                        )
                    ]
                    for label, action in reply.buttons
                ]
            )
        chunks = split_message(reply.text)
        for index, chunk in enumerate(chunks):
            await message.answer(chunk, reply_markup=markup if index == 0 else None)

    async def handle_document(self, message: Message) -> None:
        if message.from_user is None or message.document is None:
            return
        document = message.document
        file_name = document.file_name or ""
        if Path(file_name).suffix.lower() != ".txt":
            await message.answer("Поддерживаются только файлы .txt.")
            return
        if (
            document.file_size is not None
            and document.file_size > MAX_CONVERSATION_FILE_BYTES
        ):
            await message.answer(
                f"Размер файла превышает лимит "
                f"{MAX_CONVERSATION_FILE_BYTES // (1024 * 1024)} МБ."
            )
            return

        user_id = f"tg:{message.from_user.id}"
        try:
            from db.connection import SessionLocal
            from db.models import ClientOnboarding

            def waiting_for_conversation_file() -> bool:
                with SessionLocal() as db:
                    session = db.get(ClientOnboarding, user_id)
                    return session is not None and session.status == "awaiting_chat_file"

            if not await asyncio.to_thread(waiting_for_conversation_file):
                await message.answer(
                    "TXT-переписку можно загрузить только по желанию после заполнения анкеты."
                )
                return
        except Exception as error:
            logger.error("Telegram upload state lookup failed (error_type=%s)", type(error).__name__)
            await message.answer("Не удалось проверить состояние загрузки. Попробуйте позже.")
            return

        temp_path: str | None = None
        try:
            await message.answer(
                "⏳ Файл получен. Загружаю и обрабатываю переписку, пожалуйста, подождите."
            )
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
                temp_path = temp_file.name
            await message.bot.download(document, destination=temp_path)
            text = await asyncio.to_thread(load_conversation_text, temp_path)
            reply = await asyncio.to_thread(
                save_conversation_draft,
                user_id,
                text,
            )
        except ValueError as error:
            await message.answer(str(error))
            return
        except Exception as error:
            logger.error("TXT conversation import failed (error_type=%s)", type(error).__name__)
            await message.answer(
                "Не удалось обработать TXT-файл. Проверьте файл и попробуйте ещё раз."
            )
            return
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

        await self.send_reply(message, reply)

    async def handle_callback(self, callback: CallbackQuery) -> None:
        message = callback.message
        if (
            message is None
            or message.chat.type != "private"
        ):
            await callback.answer()
            return
        await callback.answer()
        action = (callback.data or "").removeprefix("onb:")
        user_id = f"tg:{callback.from_user.id}"
        try:
            reply = await asyncio.to_thread(
                self.onboarding_handler,
                user_id,
                "telegram",
                action,
            )
        except Exception as error:
            logger.error("Telegram callback failed (error_type=%s)", type(error).__name__)
            await message.answer(
                "Не удалось обработать выбор из-за технической ошибки. Попробуйте позже."
            )
            return
        if reply is not None:
            await message.edit_reply_markup(reply_markup=None)
            await self.send_reply(message, reply)

    async def handle_error(self, event: ErrorEvent) -> None:
        logger.error(
            "Telegram update failed (error_type=%s)",
            type(event.exception).__name__,
        )
        message = event.update.message if event.update is not None else None
        if message is not None:
            await message.answer(
                "Не удалось обработать сообщение из-за технической ошибки. "
                "Попробуйте позже."
            )

    def create_router(self) -> Router:
        router = Router()
        router.message.filter(lambda message: message.chat.type == "private")
        router.message.register(self.handle_document, F.document)
        router.message.register(self.handle_text_message, F.text)
        router.callback_query.register(self.handle_callback, F.data.startswith("onb:"))
        router.errors.register(self.handle_error)
        return router


async def run_polling(settings: TelegramSettings) -> None:
    bot = Bot(token=settings.token)
    dispatcher = Dispatcher()
    dispatcher.include_router(TelegramAdapter(settings).create_router())
    await dispatcher.start_polling(
        bot,
        allowed_updates=["message", "callback_query"],
    )


def main() -> None:
    load_dotenv()
    settings = load_settings()
    logger.info("Telegram-адаптер запущен")
    asyncio.run(run_polling(settings))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
