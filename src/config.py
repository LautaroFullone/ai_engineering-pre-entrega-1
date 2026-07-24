"""
Carga las API keys desde el archivo .env hacia un objeto Settings validado.

`python-dotenv` lee el .env y mete las variables en os.environ. Después las
leemos y las envolvemos en SecretStr (para que no se filtren en logs).
"""

import os

from dotenv import load_dotenv

from .schemas import Settings


def load_settings() -> Settings:
    load_dotenv()  # Lee el .env de la raíz del proyecto, si existe.

    def _wrap(name: str) -> str | None:
        value = os.getenv(name)
        return value if value else None

    return Settings(
        openai_api_key=_wrap("OPENAI_API_KEY"),
        anthropic_api_key=_wrap("ANTHROPIC_API_KEY"),
        gemini_api_key=_wrap("GEMINI_API_KEY"),
    )
