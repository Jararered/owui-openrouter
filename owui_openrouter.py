"""
title: openrouter
author(s): jararered
author_url(s): https://github.com/jararered
version: 0.3.0
"""

from typing import Dict, List, Union, Iterator
from pydantic import BaseModel, Field
import requests

class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="API key from openrouter.ai/keys",
        )
        OPENROUTER_API_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="Base URL for OpenRouter API",
        )
        OPENROUTER_PRESET: str = Field(
            default="",
            description="Preset string (e.g. @preset/lightning)",
        )
        OPENROUTER_WEB_SEARCH: bool = Field(
            default=False,
            description="Enable web search for all models (extra cost)",
        )
        SHOW_OPENROUTER_MODEL_PRICING: bool = Field(
            default=False,
            description="Show model pricing",
        )
        STRIP_OPENROUTER_STREAM_COMMENTS: bool = Field(
            default=True,
            description="Strip stream comments (e.g. : OPENROUTER PROCESSING)",
        )
        MODEL_AUTHOR_ID_WHITELIST: str = Field(
            default="",
            description="Whitelist author IDs, comma-separated (e.g. anthropic,openai)",
        )
        MODEL_AUTHOR_ID_BLACKLIST: str = Field(
            default="",
            description="Blacklist author IDs, comma-separated",
        )
        NAME_PREFIX: str = Field(
            default="",
            description="Prefix for model names (e.g. OpenRouter/google/gemma)",
        )
        APPLICATION_NAME: str = Field(
            default="OpenWebUI",
            description="Site name (X-Title header)",
        )
        APPLICATION_URL: str = Field(
            default="https://openwebui.com",
            description="Site URL (Referer header)",
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> List[Dict[str, str]]:
        if self.valves.OPENROUTER_API_KEY:
            try:
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }

                response = requests.get(
                    f"{self.valves.OPENROUTER_API_BASE_URL}/models", headers=headers
                )

                models = response.json()["data"]

                # Filter models by author whitelist and blacklist
                if self.valves.MODEL_AUTHOR_WHITELIST:
                    models = [model for model in models if model["id"].split("/")[0] in self.valves.MODEL_AUTHOR_WHITELIST.split(",")]
                if self.valves.MODEL_AUTHOR_ID_BLACKLIST:
                    models = [model for model in models if model["id"].split("/")[0] not in self.valves.MODEL_AUTHOR_ID_BLACKLIST.split(",")]

                return [
                    {
                        "id": model["id"],
                        "name": f'{self.valves.NAME_PREFIX}{model.get("name", model["id"])}',
                    }
                    for model in models
                ]

            except Exception as e:
                return [
                    {
                        "id": "Error",
                        "name": "Error fetching models. Please check your API Key.",
                    },
                ]
        else:
            return [
                {
                    "id": "Error",
                    "name": "API Key not provided.",
                },
            ]

    async def pipe(self, body: dict, __user__: dict) -> Union[str, Iterator[bytes], dict]:
        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.valves.APPLICATION_URL,
            "X-Title": self.valves.APPLICATION_NAME,
        }
        
        # Extract model id from the model name
        model_id = body["model"][body["model"].find(".") + 1 :]

        # Update the model id in the body
        payload = {**body, "model": model_id}

        if self.valves.OPENROUTER_PRESET:
            payload["preset"] = self.valves.OPENROUTER_PRESET
        if self.valves.OPENROUTER_WEB_SEARCH:
            payload["model"] = f"{model_id}:online"

        try: 
            response = requests.post(
                url=f"{self.valves.OPENROUTER_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )
            response.raise_for_status()

            if body.get("stream", False):
                if self.valves.STRIP_OPENROUTER_STREAM_COMMENTS:
                    return (self.filter_openrouter_stream_comments(line) for line in response.iter_lines() if line)
                else:
                    return response.iter_lines()
            else:
                return response.json()
        except Exception as e:
            return f"Error: {e}"

    def filter_openrouter_stream_comments(self, line: bytes) -> bytes:
        # Filter out OpenRouter processing comments (may add more here in the future)
        return line.replace(b": OPENROUTER PROCESSING", b"")