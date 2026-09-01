"""
Application configuration.

All settings are loaded from environment variables (see .env.example).
Using pydantic-settings means bad/missing config fails fast at startup
instead of causing confusing errors deep in a request.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "AI Document Search"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500"

    # --- Gemini (Google AI Studio, free tier) ---
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "gemini-2.0-flash"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"

    # --- Database ---
    # SQLite by default so the project runs with zero external services.
    # Swap this for a Postgres URL in production, e.g.:
    # postgresql+psycopg://user:pass@host:5432/dbname
    DATABASE_URL: str = "sqlite:///./rag_chatbot.db"

    # --- Vector store (Chroma, local + free, no account needed) ---
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION: str = "documents"

    # --- RAG tuning ---
    CHUNK_SIZE: int = 800          # characters per chunk
    CHUNK_OVERLAP: int = 150       # overlap between chunks
    TOP_K: int = 5                 # chunks retrieved per query
    MAX_UPLOAD_MB: int = 20

    # --- Auth ---
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60 * 24

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
