from typing import Any

from pydantic import BaseModel, field_validator

from app.config import get_settings
from query_processing.preprocess import clean


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        settings = get_settings()
        text = clean(value)
        if len(text) < settings.min_question_chars:
            raise ValueError(f"Question must be at least {settings.min_question_chars} characters long.")
        if len(text) > settings.max_question_chars:
            raise ValueError(f"Question must be at most {settings.max_question_chars} characters long.")
        if not any(ch.isalpha() for ch in text):
            raise ValueError("Question must contain words.")
        return text


def ok(data: Any) -> dict:
    return {"success": True, "data": data}


def error(code: str, message: str, details: Any = None) -> dict:
    body: dict[str, Any] = {"success": False, "error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body
