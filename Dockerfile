FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data

RUN uv sync --frozen

CMD ["uv", "run", "-m", "src.test"]
