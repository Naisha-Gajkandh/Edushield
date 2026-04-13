"""Application configuration."""
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
_DB_PATH = BASE_DIR / "student_dropout.db"


class Settings(BaseSettings):
    DATABASE_URL: str = f"sqlite+aiosqlite:///{_DB_PATH}"
    EXCEL_PATH: str = "Students_Performance_data_set.xlsx"
    ML_MODEL_PATH: str = "ml_models/risk_model.joblib"
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:8080", "http://127.0.0.1:8080",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "null",  # For file:// dashboard
    ]

    class Config:
        env_file = ".env"


settings = Settings()
