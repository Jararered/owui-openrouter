# OpenRouter Pipe for OpenWebUI

OpenWebUI pipe function that integrates with [OpenRouter](https://openrouter.ai) to provide access to 300+ AI models. Supports models from Anthropic, Google, OpenAI, Mistral AI, Meta, xAI, and other providers.

## Features

- Model filtering by author (whitelist/blacklist)
- Web search support (`:online` suffix)
- Stream comment filtering
- Preset support (`@preset/lightning`)

## Installation

1. Open OpenWebUI Settings -> Functions
2. Paste contents of `owui_openrouter.py` into the function dialog
3. Save

## Configuration

Configure via OpenWebUI Valves (Settings -> Pipes -> openrouter):

| Configuration Option               | Required | Default                 | Description                                                                                     |
| ---------------------------------- | -------- | ----------------------- | ----------------------------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY`               | Yes      | -                       | OpenRouter API key from [openrouter.ai/keys](https://openrouter.ai/keys)                        |
| `OPENROUTER_PRESET`                | Optional | -                       | Preset string (e.g., `@preset/lightning`). Applied to all requests                              |
| `OPENROUTER_WEB_SEARCH`            | Optional | `False`                 | Adds `:online` suffix to model IDs. Additional cost per OpenRouter pricing                      |
| `STRIP_OPENROUTER_STREAM_COMMENTS` | Optional | `True`                  | Filter `: OPENROUTER PROCESSING` comments from streamed responses                               |
| `AUTHOR_ID_WHITELIST`              | Optional | -                       | Comma-separated author list (e.g., `anthropic,google,openai`). Only whitelisted authors appear. |
| `AUTHOR_ID_BLACKLIST`              | Optional | -                       | Comma-separated author list. Excluded from model list. Ignored if whitelist is set              |
| `APPLICATION_NAME`                 | Optional | `OpenWebUI`             | Sent via `X-Title` header                                                                       |
| `APPLICATION_URL`                  | Optional | `https://openwebui.com` | Sent via `HTTP-Referer` header                                                                  |

## Usage

1. Set `OPENROUTER_API_KEY` in Valves
2. Configure optional settings as needed
3. Select model from OpenWebUI dropdown

Models appear with configured prefix. Author blacklist/whitelist filtering applies if configured.

Examples:

- `Google: Gemini 2.5 Pro`
- `Anthropic: Claude Sonnet 4.5`
- `Qwen: Qwen3 235B A22B`
- `Moonshot: Kimi K2 Thinking`

## Related Documentation

- [OpenWebUI Pipe Function Documentation](documentation/openwebui-pipe-function.md)
- [OpenRouter Model API Documentation](documentation/openrouter-model-api.md)
- [OpenRouter API Reference](https://openrouter.ai/docs)
