
from fastapi import FastAPI
from pydantic import BaseModel
from services.core.chat_service import chat

app = FastAPI(title="Digital Twin API")

class ChatRequest(BaseModel):
    message: str
    twin_id: int = 1
    user_id: str = "test_user"

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    answer = chat(req.message, req.twin_id, req.user_id)
    return ChatResponse(answer=answer)