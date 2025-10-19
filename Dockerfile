FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY data ./data

RUN uv sync --frozen

CMD ["uv", "run", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
