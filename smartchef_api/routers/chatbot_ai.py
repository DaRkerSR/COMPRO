from fastapi import APIRouter
from pydantic import BaseModel
import requests

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"]
)

class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chatbot_endpoint(req: ChatRequest):
    reply = ollama_chat(req.message)
    return {"reply": reply}


def ollama_chat(message: str) -> str:
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3",
        "prompt": f"""
Kamu adalah chatbot resep makanan.
Jawab dengan bahasa Indonesia yang jelas.

Pertanyaan: {message}
""",
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"