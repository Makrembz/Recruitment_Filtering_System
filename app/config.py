import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI, RateLimitError

load_dotenv()


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"
)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/cv_filter.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


@dataclass(frozen=True)
class LLMResult:
    content: str
    provider: str
    model: str


class LLMClient:
    """OpenAI-compatible client that prefers Groq and falls back on 429."""

    def __init__(self) -> None:
        self.primary_client = groq_client
        self.fallback_client = openrouter_client
        self.primary_model = GROQ_MODEL
        self.fallback_model = OPENROUTER_MODEL

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 900,
        **kwargs: Any,
    ) -> LLMResult:
        try:
            return self._complete(
                self.primary_client,
                self.primary_model,
                "groq",
                messages,
                temperature,
                max_tokens,
                **kwargs,
            )
        except (RateLimitError, APIStatusError) as exc:
            if not self._is_rate_limit(exc):
                raise
            return self._complete(
                self.fallback_client,
                self.fallback_model,
                "openrouter",
                messages,
                temperature,
                max_tokens,
                **kwargs,
            )

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        return isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429

    @staticmethod
    def _complete(
        client: OpenAI,
        model: str,
        provider: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> LLMResult:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        return LLMResult(content=content, provider=provider, model=model)


llm_client = LLMClient()
