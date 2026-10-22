import os
from dotenv import load_dotenv

load_dotenv()

import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.utils import get_random_id
from services.core.chat_service import chat

TOKEN = os.getenv("VK_TOKEN")
GROUP_ID = int(os.getenv("VK_GROUP_ID"))
TWIN_ID = 1  # проверьте, что это ID Алексея!

vk_session = vk_api.VkApi(token=TOKEN)
longpoll = VkBotLongPoll(vk_session, GROUP_ID)
vk = vk_session.get_api()

print("VK-адаптер запущен, слушаю сообщения...")

processed_ids = set()  # защита от дублей VK

for event in longpoll.listen():
    if event.type == VkBotEventType.MESSAGE_NEW:
        msg = event.obj.message
        
        # 1. Пропускаем свои же сообщения (бот не отвечает сам себе)
        if msg.get("out", 0) == 1:
            continue
        
        # 2. Защита от дублей (VK иногда присылает одно сообщение дважды)
        msg_id = msg.get("conversation_message_id") or msg.get("id")
        if msg_id in processed_ids:
            print(f"Пропущен дубль: {msg_id}")
            continue
        processed_ids.add(msg_id)
        
        user_id = str(msg["from_id"])
        text = msg.get("text", "")
        peer_id = msg["peer_id"]

        if not text:
            continue

        print(f"--- Текст: {text}, user_id: {user_id}")
        
        answer = chat(text, TWIN_ID, user_id)
        
        if answer:
            # 3. Убираем переносы, чтобы VK не разбивал на части
            answer = " ".join(answer.split()).strip()
            
            try:
                vk.messages.send(
                    peer_id=peer_id,
                    message=answer,
                    random_id=get_random_id()
                )
                print(f"Отправлено: {answer[:50]}...")
            except Exception as e:
                print(f"!!! ОШИБКА ОТПРАВКИ: {e}")