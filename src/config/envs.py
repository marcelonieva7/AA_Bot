from typing import Union
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
  QDRANT_URL: str
  QDRANT_API_KEY: Union[str, None] = None
  NVIDIA_API_KEY: str
  NVIDIA_URL: str
  EMBEDDINGS_URL: Union[str, None] = None

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
  )

settings = Settings()