"""
Esquemas Pydantic: definen y validan la estructura de los datos que entran y
salen del cliente. Son el "contrato" del sistema.
"""

from enum import Enum

from pydantic import BaseModel, Field, SecretStr


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Role(str, Enum):
    """Rol de cada mensaje en la conversación."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """
    Un mensaje individual del historial de conversación.
    Equivale a `{ role: Role; content: string }` en TS.
    """

    role: Role
    content: str


class ModelConfig(BaseModel):
    """
    Configuración validada del modelo. Acá está el corazón de la parte de
    "Validación de Esquemas" de la rúbrica.

    `Field(...)` permite poner restricciones. `ge` = greater-or-equal,
    `le` = less-or-equal. Si alguien pasa temperature=5, Pydantic lo rechaza
    ANTES de gastar un solo token llamando a la API.
    """

    provider: Provider
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=8192)


class ModelResponse(BaseModel):
    """
    Respuesta unificada. Sin importar si respondió OpenAI, Anthropic o Gemini,
    el resto de tu programa siempre recibe este mismo objeto. Ese es el punto
    de la abstracción: normalizar las diferencias entre SDKs en un solo lugar.
    """

    provider: Provider
    model: str
    content: str


class Settings(BaseModel):
    """
    Contenedor de secretos. `SecretStr` evita que las API keys aparezcan por
    accidente en logs o en un print(): '**********'

    Para leer el valor real se usa `.get_secret_value()`
    """

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
