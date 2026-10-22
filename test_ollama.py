import ollama

response = ollama.chat(
    model='qwen3:8b', 
    messages=[{'role': 'user', 'content': 'Скажи одно предложение по-русски, чтобы подтвердить, что ты работаешь.'}]
)

print(response['message']['content'])