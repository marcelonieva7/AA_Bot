import pathlib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.RAG.main import rag

TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent / 'templates'

app = FastAPI()
templates = Jinja2Templates(directory=TEMPLATES_PATH)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        response = rag(chat_request.query)
        return JSONResponse({"answer": response})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)