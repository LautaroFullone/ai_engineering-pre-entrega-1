"""
Script de validación. Prueba el cliente en modo normal y en streaming.

Cómo elegir proveedor sin tocar el código:
    python main.py                 -> usa gemini (default, tiene free tier)
    python main.py openai
    python main.py anthropic
    python main.py gemini
"""

import asyncio
import sys

from src.clients import AuthError, LLMError, RateLimitError
from src.config import load_settings
from src.manager import AsyncLLMManager
from src.schemas import ChatMessage, ModelConfig, Provider, Role

# Modelo por defecto de cada proveedor. Elegí los más baratos/rápidos para
# practicar. Podés cambiarlos por el que tengas disponible en tu cuenta.
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-3-5-haiku-latest",
    Provider.GEMINI: "gemini-2.5-flash",
}

PROMPT = "¿Qué es la entropía? Respondé en 2 o 3 frases."


async def run(provider: Provider) -> None:
    settings = load_settings()
    config = ModelConfig(
        provider=provider,
        model=DEFAULT_MODELS[provider],
        temperature=0.7,
        max_tokens=300,
    )
    manager = AsyncLLMManager(config, settings)

    messages = [
        ChatMessage(role=Role.SYSTEM, content="Sos un profesor de física conciso."),
        ChatMessage(role=Role.USER, content=PROMPT),
    ]

    print(f"\n=== Proveedor: {provider.value} | Modelo: {config.model} ===\n")

    # --- Modo normal (respuesta completa) ---
    print("[1] Modo normal (respuesta completa):")
    resp = await manager.generate(messages)
    print(resp.content)

    # --- Modo streaming (token por token) ---
    print("\n[2] Modo streaming (tokens conforme llegan):")
    async for token in manager.stream(messages):
        print(token, end="", flush=True)
    print("\n")


def main() -> None:
    # Lee el proveedor del primer argumento de la terminal; default gemini.
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "gemini"

    try:
        provider = Provider(arg)
    except ValueError:
        print(f"Proveedor desconocido: {arg}. Usá: openai | anthropic | gemini")
        sys.exit(1)

    try:
        asyncio.run(run(provider))
    except AuthError as e:
        print(f"\n[Error de credenciales] {e}")
        print("Revisá que la key correcta esté en tu .env.")
    except RateLimitError as e:
        print(f"\n[Rate limit] {e}")
        print("Esperá unos segundos y volvé a intentar.")
    except LLMError as e:
        print(f"\n[Error de LLM] {e}")


if __name__ == "__main__":
    main()
