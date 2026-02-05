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
        STRIP_OPENROUTER_STREAM_COMMENTS: bool = Field(
            default=True,
            description="Strip stream comments (e.g. : OPENROUTER PROCESSING)",
        )
        AUTHOR_ID_WHITELIST: str = Field(
            default="",
            description="Whitelist author IDs, comma-separated (e.g. anthropic,openai)",
        )
        AUTHOR_ID_BLACKLIST: str = Field(
            default="",
            description="Blacklist author IDs, comma-separated (e.g. x-ai,google)",
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

                # Get models from OpenRouter using the headers
                response = requests.get(f"{self.valves.OPENROUTER_API_BASE_URL}/models", headers=headers)
                models = response.json()["data"]

                # Filter models by author whitelist
                if self.valves.AUTHOR_ID_WHITELIST:
                    models = [
                        model
                        for model in models
                        if model["id"].split("/")[0] in self.valves.AUTHOR_ID_WHITELIST.split(",")
                    ]

                # Filter models by author blacklist
                if self.valves.AUTHOR_ID_BLACKLIST:
                    models = [
                        model
                        for model in models
                        if model["id"].split("/")[0] not in self.valves.AUTHOR_ID_BLACKLIST.split(",")
                    ]

                # Add OpenRouter prefix to model name if it's the author id of the model
                # otherwise use the model name
                pipe_models: List[Dict[str, str]] = [
                    {
                        "id": model["id"],
                        "name": f"OpenRouter: {model.get('name', model['id'])}" if model["id"].split("/")[0] == "openrouter" else model.get("name", model["id"]),
                    }
                    for model in models
                ]

                return pipe_models

            except Exception as e:
                return [{"id": "Error", "name": "API key is not valid."}]
        else:
            return [{"id": "Error", "name": "API Key not provided."}]

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

        # Add preset if provided
        if self.valves.OPENROUTER_PRESET:
            payload["preset"] = self.valves.OPENROUTER_PRESET

        # Add web search if enabled
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
