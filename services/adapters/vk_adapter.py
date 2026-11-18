import json
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import vk_api
from dotenv import load_dotenv
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.utils import get_random_id

from services.core.onboarding_service import (
    MAX_CONVERSATION_FILE_BYTES,
    OnboardingReply,
    get_twin_id,
    handle_message as handle_onboarding_message,
    load_conversation_text,
    save_conversation_draft,
)

MAX_VK_MESSAGE_LENGTH = 4000
MAX_PROCESSED_EVENTS = 10_000
logger = logging.getLogger(__name__)


def _chunks(text: str, limit: int = MAX_VK_MESSAGE_LENGTH) -> list[str]:
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


def _file_url_is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.port not in (None, 443):
        return False
    host = parsed.hostname.lower()
    return any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in (".vk.com", ".vk.me", ".userapi.com")
    )


def _download_vk_file(url: str, destination: str) -> None:
    if not _file_url_is_allowed(url):
        raise ValueError("Недопустимый адрес файла VK")

    total = 0
    with requests.get(
        url,
        stream=True,
        timeout=(5, 30),
        allow_redirects=False,
    ) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_CONVERSATION_FILE_BYTES:
            raise ValueError(
                f"Файл превышает лимит "
                f"{MAX_CONVERSATION_FILE_BYTES // (1024 * 1024)} МБ"
            )
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_CONVERSATION_FILE_BYTES:
                    raise ValueError(
                        f"Файл превышает лимит "
                        f"{MAX_CONVERSATION_FILE_BYTES // (1024 * 1024)} МБ"
                    )
                file.write(chunk)


class VkAdapter:
    def __init__(
        self,
        api,
        onboarding_handler=handle_onboarding_message,
        twin_lookup=get_twin_id,
        chat_handler=None,
        file_downloader=_download_vk_file,
        text_loader=load_conversation_text,
        text_saver=save_conversation_draft,
    ) -> None:
        self.api = api
        self.onboarding_handler = onboarding_handler
        self.twin_lookup = twin_lookup
        self.chat_handler = chat_handler
        self.file_downloader = file_downloader
        self.text_loader = text_loader
        self.text_saver = text_saver
        self.processed_ids: set[tuple[int, int]] = set()

    @staticmethod
    def _keyboard(reply: OnboardingReply) -> str | None:
        if not reply.buttons:
            return json.dumps({"one_time": True, "buttons": []})
        keyboard = {
            "one_time": False,
            "buttons": [
                [
                    {
                        "action": {
                            "type": "text",
                            "label": label[:40],
                            "payload": json.dumps({"action": action}),
                        },
                        "color": "positive" if "yes" in action or "confirm" in action else "secondary",
                    }
                ]
                for label, action in reply.buttons
            ],
        }
        return json.dumps(keyboard, ensure_ascii=False)

    def _send(self, peer_id: int, text: str, reply: OnboardingReply | None = None) -> None:
        chunks = _chunks(text)
        keyboard = self._keyboard(reply) if reply is not None else None
        for index, chunk in enumerate(chunks):
            self.api.messages.send(
                peer_id=peer_id,
                message=chunk,
                random_id=get_random_id(),
                keyboard=keyboard if index == 0 else None,
            )

    def _handle_conversation_file(
        self,
        user_id: str,
        peer_id: int,
        attachments: list[dict],
    ) -> bool:
        uploaded_file = next(
            (item.get("doc", {}) for item in attachments if item.get("type") == "doc"),
            None,
        )
        if uploaded_file is None:
            return False
        try:
            from db.connection import SessionLocal
            from db.models import ClientOnboarding

            with SessionLocal() as db:
                onboarding = db.get(ClientOnboarding, user_id)
                waiting_for_file = (
                    onboarding is not None and onboarding.status == "awaiting_chat_file"
                )
        except Exception as error:
            logger.error("VK upload state lookup failed (error_type=%s)", type(error).__name__)
            self._send(peer_id, "Не удалось проверить состояние загрузки. Попробуйте позже.")
            return True
        if not waiting_for_file:
            self._send(
                peer_id,
                "TXT-переписку можно загрузить только по желанию после заполнения анкеты.",
            )
            return True

        if str(uploaded_file.get("ext", "")).lower().lstrip(".") != "txt":
            self._send(peer_id, "Поддерживаются только файлы .txt.")
            return True
        try:
            size = int(uploaded_file.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        if size > MAX_CONVERSATION_FILE_BYTES:
            self._send(
                peer_id,
                f"Размер файла превышает лимит "
                f"{MAX_CONVERSATION_FILE_BYTES // (1024 * 1024)} МБ.",
            )
            return True

        temp_path: str | None = None
        try:
            self._send(
                peer_id,
                "⏳ Файл получен. Загружаю и обрабатываю переписку, пожалуйста, подождите.",
            )
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
                temp_path = temp_file.name
            url = uploaded_file.get("url")
            if not isinstance(url, str):
                raise ValueError("Вложение не содержит ссылку на TXT-файл")
            self.file_downloader(url, temp_path)
            content = self.text_loader(temp_path)
            reply = self.text_saver(user_id, content)
            self._send(peer_id, reply.text, reply)
        except ValueError as error:
            self._send(peer_id, str(error))
        except Exception as error:
            logger.error("VK TXT conversation import failed (error_type=%s)", type(error).__name__)
            self._send(
                peer_id,
                "Не удалось обработать TXT-файл. Проверьте файл и попробуйте ещё раз.",
            )
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)
        return True

    def handle_message(self, message: dict) -> None:
        if message.get("out", 0) == 1:
            return

        try:
            numeric_user_id = int(message["from_id"])
            peer_id = int(message["peer_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning("Skipped VK message with invalid sender or peer")
            return

        if peer_id != numeric_user_id:
            return

        try:
            message_id = int(
                message.get("conversation_message_id") or message.get("id") or 0
            )
        except (TypeError, ValueError):
            logger.warning("Skipped VK message with invalid message ID")
            return
        event_id = (peer_id, message_id)
        if event_id in self.processed_ids:
            return
        if len(self.processed_ids) >= MAX_PROCESSED_EVENTS:
            self.processed_ids.clear()
        self.processed_ids.add(event_id)

        user_id = f"vk:{numeric_user_id}"
        attachments = message.get("attachments") or []
        if attachments and self._handle_conversation_file(user_id, peer_id, attachments):
            return

        payload = message.get("payload")
        action = None
        if isinstance(payload, dict):
            action = payload.get("action")
        elif isinstance(payload, str):
            try:
                action = json.loads(payload).get("action")
            except (json.JSONDecodeError, AttributeError):
                action = None
        text = str(action or message.get("text") or "").strip()
        if not text:
            return

        try:
            onboarding_reply = self.onboarding_handler(user_id, "vk", text)
            if onboarding_reply is not None:
                self._send(peer_id, onboarding_reply.text, onboarding_reply)
                return

            twin_id = self.twin_lookup(user_id)
            if twin_id is None:
                self._send(peer_id, "Профиль не найден. Чтобы начать заполнение, отправьте /start.")
                return

            chat_handler = self.chat_handler
            if chat_handler is None:
                from services.core.chat_service import chat

                chat_handler = chat
            answer = chat_handler(text, twin_id, user_id)
            if answer:
                self._send(peer_id, answer)
        except Exception as error:
            logger.error("VK message processing failed (error_type=%s)", type(error).__name__)
            self._send(peer_id, "Не удалось обработать сообщение из-за технической ошибки. Попробуйте позже.")


def _load_settings() -> tuple[str, int]:
    token = os.getenv("VK_TOKEN", "").strip()
    if not token:
        raise ValueError("VK_TOKEN is required")
    try:
        group_id = int(os.getenv("VK_GROUP_ID", ""))
    except ValueError:
        raise ValueError("VK_GROUP_ID must be a positive integer") from None
    if group_id <= 0:
        raise ValueError("VK_GROUP_ID must be a positive integer")

    return token, group_id


def main() -> None:
    load_dotenv()
    token, group_id = _load_settings()
    session = vk_api.VkApi(token=token)
    api = session.get_api()
    adapter = VkAdapter(api)
    longpoll = VkBotLongPoll(session, group_id)

    logger.info("VK-адаптер запущен")
    for event in longpoll.listen():
        if event.type != VkBotEventType.MESSAGE_NEW:
            continue
        adapter.handle_message(event.obj.message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
