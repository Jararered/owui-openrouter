"""
title: openrouter
author(s): jararered + gemini-3-pro
author_url(s): https://github.com/jararered/owui-openrouter + https://deepmind.google/models/gemini/pro/
version: 0.1.0
"""

from pydantic import BaseModel, Field
import requests
from typing import List, Union, Iterator


class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your OpenRouter API Key. Get one at openrouter.ai/keys",
        )
        NAME_PREFIX: str = Field(
            default="",
            description="Prefix to be added before model names (e.g., OpenRouter/google/gemma).",
        )
        YOUR_SITE_URL: str = Field(
            default="https://openwebui.com",
            description="Optional: Your site URL for OpenRouter rankings (HTTP-Referer header).",
        )
        YOUR_SITE_NAME: str = Field(
            default="OpenWebUI",
            description="Optional: Your site name for OpenRouter rankings (X-Title header).",
        )
        MODEL_AUTHORS: str = Field(
            default="openai,anthropic,google,mistral,meta",
            description="Optional: Comma-separated list of model authors to filter by.",
        )
        SHOW_PRICING: bool = Field(
            default=True,
            description="Optional: Whether to show pricing information for the models.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.api_base = "https://openrouter.ai/api/v1"

    def pipes(self) -> List[dict]:
        """
        Fetches the list of available models from OpenRouter.
        """
        if not self.valves.OPENROUTER_API_KEY:
            return [
                {
                    "id": "error",
                    "name": "Error: OpenRouter API Key not set in Valves.",
                }
            ]

        try:
            headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            }

            # Fetch models from OpenRouter
            r = requests.get(f"{self.api_base}/models", headers=headers)
            r.raise_for_status()
            models_data = r.json()

            # Filter models by authors if provided
            if self.valves.MODEL_AUTHORS:
                models_data["data"] = [
                    model
                    for model in models_data["data"]
                    if model["id"].split("/")[0] in self.valves.MODEL_AUTHORS.split(",")
                ]

            # Sort the models alphabetically by id
            models_data["data"].sort(key=lambda x: x["id"])

            # Append pricing information to the models if not free model
            if self.valves.SHOW_PRICING and not "free" in [model["id"] for model in models_data["data"]]:
                for model in models_data["data"]:
                    promptPerMillion = float(model['pricing']['prompt']) * 1000000
                    completionPerMillion = float(model['pricing']['completion']) * 1000000
                    model["pricing"] = [
                        {
                            "prompt": f"${promptPerMillion:.4f}/m in",
                            "completion": f"${completionPerMillion:.4f}/m out",
                        }
                    ]

            # Transform OpenRouter models to OpenWebUI format
            # We map OpenRouter 'id' to both id and name, pre-pending the user's chosen prefix
            return [
                {
                    "id": model["id"],
                    "name": f"{self.valves.NAME_PREFIX}{model.get('id') } - {model.get('pricing', {}).get('prompt', '0')}",
                }
                for model in models_data.get("data", [])
            ]

        except Exception as e:
            return [
                {
                    "id": "error",
                    "name": f"Error fetching models: {str(e)}",
                }
            ]

    def pipe(self, body: dict, __user__: dict) -> Union[str, Iterator[str]]:
        """
        Handles the chat completion request to OpenRouter.
        """
        print(f"Pipe called for model: {body.get('model')}")

        headers = {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.valves.YOUR_SITE_URL,
            "X-Title": self.valves.YOUR_SITE_NAME,
        }

        # The model ID usually comes in as "function_file_name.model_id"
        # We need to strip the prefix to get the actual OpenRouter model ID
        # Example: "openrouter_pipe.google/gemma-7b" -> "google/gemma-7b"
        if "." in body["model"]:
            model_id = body["model"][body["model"].find(".") + 1 :]
        else:
            model_id = body["model"]

        # Update the body with the clean model ID
        payload = {**body, "model": model_id}

        try:
            r = requests.post(
                url=f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()

        except Exception as e:
            return f"Error: {e}"
