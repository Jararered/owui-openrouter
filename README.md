# OpenRouter Pipe for OpenWebUI

OpenWebUI pipe that integrates with [OpenRouter](https://openrouter.ai) to provide access to 400+ AI models. Supports models from Anthropic, Google, OpenAI, Mistral AI, Meta, xAI, and other providers.

## Features

- Model filtering by author (whitelist/blacklist)
- Web search support (`:online` suffix)
- Optional pricing display in model selector
- Stream comment filtering
- Preset support (`@preset/lightning`)
- Custom model name prefixes
- Reusable `OpenrouterAPI` client class

## Installation

**Option 1: File system**

1. Copy `owui_openrouter.py` to OpenWebUI pipes directory:
   ```bash
   cp owui_openrouter.py /path/to/open-webui/data/pipes/
   ```
2. Restart OpenWebUI

**Option 2: Function dialog**

1. Open OpenWebUI Settings → Functions
2. Paste contents of `owui_openrouter.py` into the function dialog
3. Save

## Configuration

Configure via OpenWebUI Valves (Settings -> Pipes -> openrouter):

| Configuration Option               | Required | Default                 | Description                                                                                                         |
| ---------------------------------- | -------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY`               | Yes      | -                       | OpenRouter API key from [openrouter.ai/keys](https://openrouter.ai/keys)                                            |
| `OPENROUTER_PRESET`                | Optional | -                       | Preset string (e.g., `@preset/lightning`). Applied to all requests                                                  |
| `OPENROUTER_WEB_SEARCH`            | Optional | `False`                 | Adds `:online` suffix to model IDs. Additional cost per OpenRouter pricing                                          |
| `SHOW_OPENROUTER_MODEL_PRICING`    | Optional | `False`                 | Display pricing as `$X/m in - $Y/m out` (per million tokens). Shows "free" for $0 models                            |
| `STRIP_OPENROUTER_STREAM_COMMENTS` | Optional | `True`                  | Filter `: OPENROUTER PROCESSING` comments from streamed responses                                                   |
| `MODEL_AUTHOR_WHITELIST`           | Optional | -                       | Comma-separated author list (e.g., `anthropic,google,openai`). Only whitelisted authors appear. Overrides blacklist |
| `MODEL_AUTHOR_BLACKLIST`           | Optional | -                       | Comma-separated author list. Excluded from model list. Ignored if whitelist is set                                  |
| `NAME_PREFIX`                      | Optional | -                       | Prefix for model names (e.g., `OpenRouter/` → `OpenRouter/google/gemma-7b`)                                         |
| `APPLICATION_NAME`                 | Optional | `OpenWebUI`             | Sent via `X-Title` header                                                                                           |
| `APPLICATION_URL`                  | Optional | `https://openwebui.com` | Sent via `HTTP-Referer` header                                                                                      |

## Usage

1. Set `OPENROUTER_API_KEY` in Valves
2. Configure optional settings as needed
3. Select model from OpenWebUI dropdown

Models appear with configured prefix and pricing (if enabled). Author blacklist/whitelist filtering applies if configured.

Examples:

- `google/gemini-2.5-pro`
- `OpenRouter/google/gemini-2.5-pro` (with prefix)
- `google/gemini-2.5-pro ($0.05/m in - $0.05/m out)` (with pricing)

## Related Documentation

- [OpenWebUI Pipe Function Documentation](documentation/openwebui-pipe-function.md)
- [OpenRouter Model API Documentation](documentation/openrouter-model-api.md)
- [OpenRouter API Reference](https://openrouter.ai/docs)
