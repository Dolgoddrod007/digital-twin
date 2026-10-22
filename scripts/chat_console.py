from services.core.chat_service import chat

TWIN_ID = 4          # фактический ID Алексея в вашей базе
USER_ID = "console_user"

if __name__ == "__main__":
    print("Консольный чат с цифровым двойником. Введите 'exit' для выхода.\n")
    while True:
        user = input("Вы: ")
        if user.lower() in ("exit", "quit"):
            break
        answer = chat(user, TWIN_ID, USER_ID)
        print(f"Двойник: {answer}\n")