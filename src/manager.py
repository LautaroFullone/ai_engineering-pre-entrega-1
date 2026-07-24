"""
El AsyncLLMManager: la pieza que la rúbrica valora en un 35%.

Es la ÚNICA clase que tu lógica de negocio necesita conocer. Vos le pasás una
config (qué proveedor, qué modelo) y él se encarga de instanciar el cliente
correcto por debajo (Factory Pattern). Cambiar de OpenAI a Gemini es cambiar
un string, no reescribir código.
"""

from collections.abc import AsyncIterator

from .clients import (
    AnthropicClient,
    AuthError,
    BaseLLMClient,
    GeminiClient,
    OpenAIClient,
)
from .schemas import ChatMessage, ModelConfig, ModelResponse, Provider, Settings

# El "registro" del Factory: mapea cada proveedor a su clase.
# Agregar un proveedor nuevo = agregar una línea acá + su clase. Nada más.
_REGISTRY: dict[Provider, type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}


class AsyncLLMManager:
    def __init__(self, config: ModelConfig, settings: Settings) -> None:
        self.config = config
        self._client = self._build_client(config, settings)

    def _build_client(self, config: ModelConfig, settings: Settings) -> BaseLLMClient:
        """El Factory: elige y construye el cliente correcto según el provider."""
        key_map = {
            Provider.OPENAI: settings.openai_api_key,
            Provider.ANTHROPIC: settings.anthropic_api_key,
            Provider.GEMINI: settings.gemini_api_key,
        }

        secret = key_map[config.provider]

        if secret is None:
            raise AuthError(
                f"Falta la API key para {config.provider.value}. "
                f"Definila en tu archivo .env."
            )

        client_cls = _REGISTRY[config.provider]
        # .get_secret_value() extrae la key real del SecretStr, solo en este punto.
        return client_cls(config, secret.get_secret_value())

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Delega en el cliente concreto."""
        return await self._client.generate(messages)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """
        Streaming.
        Como este método tiene `yield` por debajo (a través del cliente), es un
        generador asíncrono. Por eso hacemos `async for ... yield` en vez de
        `return`. Si pones `return self._client.stream(...)` acá también
        funcionaría, pero re-emitir con async for deja el patrón explícito.
        """
        async for token in self._client.stream(messages):
            yield token
