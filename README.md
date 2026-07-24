# Unified Async LLM Client

Pre-entrega 1 — Módulo 1 (AI Engineering, Coderhouse).

Capa de abstracción asíncrona que permite hablar con **OpenAI**, **Anthropic** y
**Gemini** a través de una interfaz única. Cambiar de proveedor es cambiar un
string de configuración, sin tocar la lógica de negocio.

## Qué implementa

- **Intercambiabilidad de proveedores** mediante un Factory Pattern (`AsyncLLMManager`).
- **Asincronía total**: todas las llamadas usan `async`/`await`, sin bloquear el event loop.
- **Streaming** de tokens mediante generadores asíncronos (`async for` + `yield`).
- **Validación con Pydantic**: esquemas de mensajes y configuración, con rangos
  validados (ej. `temperature` entre 0 y 2) antes de gastar tokens.
- **Manejo de errores resiliente**: los fallos de red, autenticación y rate
  limit se capturan y se traducen a errores propios controlados, en vez de
  romper el programa.
- **Secretos seguros**: las API keys se manejan con `SecretStr` y se cargan
  desde un `.env` que nunca se sube a git.

## Estructura

```
ia_engineering-pre-entrega-1/
├── src/
│   ├── schemas.py    # Modelos Pydantic (mensajes, config, respuesta, secretos)
│   ├── clients.py    # BaseLLMClient abstracto + OpenAI / Anthropic / Gemini
│   ├── manager.py    # AsyncLLMManager (Factory Pattern)
│   └── config.py     # Carga de settings desde .env
├── main.py           # Script de validación (modo normal + streaming)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Requisitos

- **Python 3.12+**
- Al menos una API key de un proveedor.

## Instalación (macOS / zsh)

```zsh
# 1. Crear un entorno virtual aislado para este proyecto
python3.12 -m venv .venv

# 2. Activarlo (verás (.venv) al inicio del prompt)
source .venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Crear tu archivo de secretos a partir de la plantilla
cp .env.example .env
```

Después abrí `.env` y pegá tu(s) key(s) en la línea correspondiente.

## Uso

```zsh
python main.py            # usa Gemini (default, gratis)
python main.py openai     # usa OpenAI
python main.py anthropic  # usa Anthropic
python main.py gemini     # usa Gemini
```

El script hace la misma pregunta ("¿Qué es la entropía?") en **modo normal** y
en **modo streaming**, para verificar ambos caminos.

## Variables de entorno

| Variable            | Requerida           | Cómo obtenerla                                               |
| ------------------- | ------------------- | ------------------------------------------------------------ |
| `GEMINI_API_KEY`    | Para usar Gemini    | https://aistudio.google.com/apikey (free tier)               |
| `OPENAI_API_KEY`    | Para usar OpenAI    | https://platform.openai.com/api-keys (requiere saldo)        |
| `ANTHROPIC_API_KEY` | Para usar Anthropic | https://console.anthropic.com/settings/keys (requiere saldo) |

Solo necesitás la key del proveedor que vayas a probar.

## Cómo extender

Agregar un proveedor nuevo son 2 pasos:

1. Crear una clase que herede de `BaseLLMClient` en `clients.py` e implemente
   `generate()` y `stream()`.
2. Registrarla en el diccionario `_REGISTRY` de `manager.py`.

El resto del sistema no se entera del cambio.
