"""
Los clientes: una clase base abstracta que define el "contrato" (qué métodos
DEBE tener todo cliente) y tres implementaciones concretas.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Role


# --------------------------------------------------------------------------- #
# Errores propios: capturamos los fallos del SDK y los "traducimos" a un error
# nuestro y controlado. Así el resto del programa no necesita saber nada de
# las excepciones internas de cada SDK (que son todas distintas).
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """Error base de nuestra capa de cliente."""


class AuthError(LLMError):
    """API key faltante o inválida."""


class RateLimitError(LLMError):
    """Límite de tasa alcanzado (demasiadas requests)."""


def _split_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, list[ChatMessage]]:
    """
    Helper: separa el mensaje de sistema del resto.

    Por qué existe: OpenAI mete el system dentro de la lista de mensajes, pero
    Anthropic y Gemini lo reciben como un parámetro APARTE. Esta función unifica
    esa diferencia para no repetir lógica en cada cliente.
    """
    system: str | None = None
    rest: list[ChatMessage] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            system = m.content
        else:
            rest.append(m)
    return system, rest


# --------------------------------------------------------------------------- #
# Clase base abstracta
# --------------------------------------------------------------------------- #
class BaseLLMClient(ABC):
    """
    Contrato común. Todo cliente concreto recibe una `ModelConfig` y su API key,
    y debe saber responder de forma normal (`generate`) y en streaming (`stream`).
    """

    def __init__(self, config: ModelConfig, api_key: str) -> None:
        self.config = config
        self.api_key = api_key

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Devuelve la respuesta completa de una vez."""
        ...

    @abstractmethod
    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """
        Devuelve un generador asíncrono de fragmentos de texto (tokens).
        """
        ...


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config, api_key)
        # Import local: solo cargamos el SDK cuando realmente se usa este cliente.
        # Así, si solo probás con Gemini, no necesitás tener instalado openai.
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)

    def _to_openai(self, messages: list[ChatMessage]) -> list[dict]:
        # OpenAI espera dicts con role/content, y acepta "system" en la lista.
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        from openai import APIError, AuthenticationError, RateLimitError as OAIRate

        try:
            resp = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self._to_openai(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except AuthenticationError as e:
            raise AuthError(f"OpenAI: credenciales inválidas ({e})") from e
        except OAIRate as e:
            raise RateLimitError(f"OpenAI: rate limit ({e})") from e
        except APIError as e:
            raise LLMError(f"OpenAI: error de API ({e})") from e

        return ModelResponse(
            provider=Provider.OPENAI,
            model=self.config.model,
            content=resp.choices[0].message.content or "",
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        from openai import APIError, AuthenticationError, RateLimitError as OAIRate

        try:
            stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self._to_openai(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except AuthenticationError as e:
            raise AuthError(f"OpenAI: credenciales inválidas ({e})") from e
        except OAIRate as e:
            raise RateLimitError(f"OpenAI: rate limit ({e})") from e
        except APIError as e:
            raise LLMError(f"OpenAI: error de API ({e})") from e


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config, api_key)
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        from anthropic import APIError, AuthenticationError, RateLimitError as AntRate

        system, rest = _split_messages(messages)
        # Anthropic: el system va como parámetro aparte, no dentro de messages.
        try:
            resp = await self._client.messages.create(
                model=self.config.model,
                system=system or "",
                messages=[{"role": m.role.value, "content": m.content} for m in rest],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except AuthenticationError as e:
            raise AuthError(f"Anthropic: credenciales inválidas ({e})") from e
        except AntRate as e:
            raise RateLimitError(f"Anthropic: rate limit ({e})") from e
        except APIError as e:
            raise LLMError(f"Anthropic: error de API ({e})") from e

        # Anthropic devuelve content[0].text (distinto a OpenAI). Lo normalizamos.
        return ModelResponse(
            provider=Provider.ANTHROPIC,
            model=self.config.model,
            content=resp.content[0].text if resp.content else "",
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        from anthropic import APIError, AuthenticationError, RateLimitError as AntRate

        system, rest = _split_messages(messages)
        try:
            async with self._client.messages.stream(
                model=self.config.model,
                system=system or "",
                messages=[{"role": m.role.value, "content": m.content} for m in rest],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except AuthenticationError as e:
            raise AuthError(f"Anthropic: credenciales inválidas ({e})") from e
        except AntRate as e:
            raise RateLimitError(f"Anthropic: rate limit ({e})") from e
        except APIError as e:
            raise LLMError(f"Anthropic: error de API ({e})") from e


# --------------------------------------------------------------------------- #
# Gemini (google-genai)
# --------------------------------------------------------------------------- #
class GeminiClient(BaseLLMClient):
    def __init__(self, config: ModelConfig, api_key: str) -> None:
        super().__init__(config, api_key)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def _to_gemini(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        # Gemini usa "model" en vez de "assistant", y el system va aparte.
        system, rest = _split_messages(messages)
        contents = []
        for m in rest:
            role = "model" if m.role == Role.ASSISTANT else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        return system, contents

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        from google.genai import types
        from google.genai import errors as gerrors

        system, contents = self._to_gemini(messages)
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
            )
        except gerrors.ClientError as e:
            # Gemini agrupa varios errores acá; distinguimos por código HTTP.
            if getattr(e, "code", None) == 429:
                raise RateLimitError(f"Gemini: rate limit ({e})") from e
            if getattr(e, "code", None) in (401, 403):
                raise AuthError(f"Gemini: credenciales inválidas ({e})") from e
            raise LLMError(f"Gemini: error de API ({e})") from e

        return ModelResponse(
            provider=Provider.GEMINI,
            model=self.config.model,
            content=resp.text or "",
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        from google.genai import types
        from google.genai import errors as gerrors

        system, contents = self._to_gemini(messages)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                ),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except gerrors.ClientError as e:
            if getattr(e, "code", None) == 429:
                raise RateLimitError(f"Gemini: rate limit ({e})") from e
            if getattr(e, "code", None) in (401, 403):
                raise AuthError(f"Gemini: credenciales inválidas ({e})") from e
            raise LLMError(f"Gemini: error de API ({e})") from e
