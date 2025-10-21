import pathlib
import logging
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["ONNXRUNTIME_LOG_LEVEL"] = "4"

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from src.monitoring.tracing import tracer

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
TEMPLATES_PATH = pathlib.Path(__file__).resolve().parent / 'templates'

# FastAPI app
app = FastAPI(title="AA Bot RAG")
templates = Jinja2Templates(directory=TEMPLATES_PATH)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    query: str
    model: str = 'meta/llama-4-scout-17b-16e-instruct'

# Startup
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting AA Bot RAG...")
    try:
        # Pre-cargar modelos (opcional pero recomendado)
        from src.config.db import qdrant_db
        # Test connection
        collections = qdrant_db.client.get_collections()
        logger.info(f"✅ Connected to Qdrant: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.warning(f"⚠️ Startup warning: {e}")

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    try:
        from src.config.db import qdrant_db
        collections = qdrant_db.client.get_collections()
        
        return {
            "status": "healthy",
            "services": {
                "qdrant": True,
                "sparse_model": qdrant_db._sparse_model is not None
            },
            "collections": [c.name for c in collections.collections]
        }
    except Exception as e:
        return JSONResponse(
            {"status": "degraded", "error": str(e)},
            status_code=503
        )

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest, request: Request):
    with tracer.start_as_current_span("api.chat") as span:
        if request.client:
            span.set_attribute("request.client_ip", request.client.host)
        span.set_attribute("chat.model", chat_request.model)
        span.set_attribute("chat.query_preview", chat_request.query[:200])
        span.set_attribute("chat.query_length", len(chat_request.query))
        try:
            from src.RAG.main import rag
            logger.info(f"📩 Query: {chat_request.query[:50]}...")
            response = rag(
                query=chat_request.query,
                model=chat_request.model
            )
            span.set_attribute("chat.response_preview", response[:200])
            span.set_status(Status(StatusCode.OK))
            logger.info("✅ Response generated")

            return JSONResponse({"answer": response})
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))

            return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/collections")
async def list_collections():
    """Debug endpoint para ver colecciones"""
    try:
        from src.config.db import qdrant_db
        collections = qdrant_db.client.get_collections()
        return {
            "collections": [
                {
                    "name": c.name,
                    "points_count": qdrant_db.client.count(c.name).count
                }
                for c in collections.collections
            ]
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)