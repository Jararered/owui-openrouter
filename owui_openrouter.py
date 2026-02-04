"""
title: openrouter
author(s): jararered
author_url(s): https://github.com/jararered
version: 0.2.0
"""



from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Union, Iterator

import requests
from pydantic import BaseModel, Field


def ErrorModel(message: str) -> dict:
    """Creates an error model dictionary for display in OpenWebUI."""
    return {
        "id": "error",
        "name": message,
    }


class Pipe:
    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="API key from openrouter.ai/keys",
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
        MODEL_AUTHOR_WHITELIST: str = Field(
            default="",
            description="Whitelist authors, comma-separated (e.g. anthropic,openai)",
        )
        MODEL_AUTHOR_BLACKLIST: str = Field(
            default="",
            description="Blacklist authors, comma-separated",
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
        config = OpenrouterAPIConfig(
            api_key=self.valves.OPENROUTER_API_KEY,
            application_url=self.valves.APPLICATION_URL,
            application_name=self.valves.APPLICATION_NAME,
            model_author_whitelist=self.valves.MODEL_AUTHOR_WHITELIST,
            model_author_blacklist=self.valves.MODEL_AUTHOR_BLACKLIST,
            show_model_pricing=self.valves.SHOW_OPENROUTER_MODEL_PRICING,
            name_prefix=self.valves.NAME_PREFIX,
        )
        self.api = OpenrouterAPI(config)

    def filter_stream_line(self, line: bytes) -> bytes:
        """
        Filters out stream comments from SSE response lines.

        This is an abstract filter function that can be extended to filter
        different types of stream content. Needs to be bytes to be compatible
        with the OpenWebUI SSE spec.
        """
        if not self.valves.STRIP_OPENROUTER_STREAM_COMMENTS:
            return line

        # Filter out OpenRouter processing comments
        filtered = line.replace(b": OPENROUTER PROCESSING", b"")
        return filtered

    def pipes(self) -> List[dict]:
        """
        Fetches the list of available models from OpenRouter.
        Returns a list of model dictionaries in OpenWebUI format.
        """
        return self.api.get_models()

    def pipe(self, body: dict, __user__: dict) -> Union[str, Iterator[bytes], dict]:
        """
        Handles the chat completion request to OpenRouter.

        This is the main pipe function that processes requests and returns
        either streaming responses or complete JSON responses.
        """
        print(f"Pipe called for model: {body.get('model')}")

        stream = body.get("stream", False)
        result = self.api.chat_completion(
            body,
            stream=stream,
            preset=self.valves.OPENROUTER_PRESET,
            web_search=self.valves.OPENROUTER_WEB_SEARCH,
        )

        if stream and not isinstance(result, dict):
            return (self.filter_stream_line(line) for line in result)
        return result



@dataclass
class OpenrouterAPIConfig:
    """Configuration for OpenrouterAPI. Pass values from your app's settings."""

    api_key: str = ""
    application_url: str = "https://openwebui.com"
    application_name: str = "OpenWebUI"
    model_author_whitelist: str = ""
    model_author_blacklist: str = ""
    show_model_pricing: bool = False
    name_prefix: str = ""



class OpenrouterAPI:
    """
    Client for the OpenRouter API. Use this class in other files to list models
    and perform chat completions without depending on OpenWebUI Pipe.
    """

    def __init__(self, config: OpenrouterAPIConfig):
        self.config = config
        self.api_base = "https://openrouter.ai/api/v1"

    def get_auth_headers(self) -> Dict[str, str]:
        """Creates authentication headers for API requests."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def get_request_headers(self) -> Dict[str, str]:
        """Creates complete headers for chat completion requests."""
        headers = self.get_auth_headers()
        headers.update(
            {
                "HTTP-Referer": self.config.application_url,
                "X-Title": self.config.application_name,
            }
        )
        return headers

    def extract_model_id(self, model_string: str) -> str:
        """
        Extracts the actual OpenRouter model ID from the model string.

        The model ID usually comes in as "function_file_name.model_id"
        Example: "openrouter_pipe.google/gemma-7b" -> "google/gemma-7b"
        """
        if "." in model_string:
            return model_string[model_string.find(".") + 1 :]
        return model_string

    def apply_author_whitelist(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters models by the specified authors."""
        if not self.config.model_author_whitelist:
            return models
        authors = [
            author.strip()
            for author in self.config.model_author_whitelist.split(",")
        ]
        return [m for m in models if m["id"].split("/")[0] in authors]

    def apply_author_blacklist(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filters models by the specified authors."""
        if not self.config.model_author_blacklist:
            return models
        authors = [
            author.strip()
            for author in self.config.model_author_blacklist.split(",")
        ]
        return [m for m in models if m["id"].split("/")[0] not in authors]

    def format_pricing_string(self, prompt_price: str, completion_price: str) -> str:
        """Formats pricing information as a display string."""
        if prompt_price == "0" and completion_price == "0":
            return "free"
        return f"${prompt_price}/m in - ${completion_price}/m out"

    def format_price(self, price_per_token: float) -> str:
        """
        Formats a price per token to a price per million tokens.
        - Rounds repeating 9's up (e.g., 2.999 -> 3.0)
        - Displays whole numbers as integers (e.g., 3.0 -> 3)
        - Displays decimals with up to 3 decimal places
        """
        price_per_million = Decimal(str(price_per_token)) * Decimal("1000000")
        rounded = price_per_million.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        if rounded == rounded.to_integral_value():
            return str(int(rounded))
        formatted = str(rounded)
        return formatted.rstrip("0").rstrip(".")

    def format_model_pricing(self, model: Dict[str, Any]) -> str:
        """Formats pricing information for a model."""
        prompt_price = self.format_price(float(model["pricing"]["prompt"]))
        completion_price = self.format_price(float(model["pricing"]["completion"]))
        return self.format_pricing_string(prompt_price, completion_price)

    def transform_model_to_openwebui_format(self, model: Dict[str, Any]) -> Dict[str, str]:
        """Transforms an OpenRouter model to OpenWebUI format."""
        pricing_display = (
            self.format_model_pricing(model)
            if self.config.show_model_pricing
            else ""
        )
        name = f"{self.config.name_prefix}{model['id']}"
        if pricing_display:
            name = f"{name} ({pricing_display})"
        return {"id": model["id"], "name": name}

    def handle_api_error(self, response: requests.Response) -> Dict[str, Any]:
        """Handles API error responses from OpenRouter."""
        try:
            return response.json()
        except (ValueError, requests.exceptions.JSONDecodeError):
            return {
                "error": {
                    "code": response.status_code,
                    "message": f"HTTP {response.status_code}: {response.reason}",
                }
            }

    def handle_request_exception(self, exception: Exception) -> Dict[str, Any]:
        """Handles request exceptions with appropriate error formatting."""
        if isinstance(exception, requests.exceptions.RequestException):
            return {"error": {"code": "request_error", "message": str(exception)}}
        return {"error": {"code": "unknown_error", "message": str(exception)}}

    def get_models(self) -> List[Dict[str, str]]:
        """
        Fetches the list of available models from OpenRouter.
        Returns a list of model dictionaries in OpenWebUI format.
        """
        if not self.config.api_key:
            return [{"id": "error", "name": "Error: OpenRouter API Key not set in Valves."}]

        try:
            headers = self.get_auth_headers()
            response = requests.get(f"{self.api_base}/models", headers=headers)
            response.raise_for_status()

            model_data = response.json()["data"]
            model_data.sort(key=lambda m: m["id"])
            model_data = self.apply_author_whitelist(model_data)
            model_data = self.apply_author_blacklist(model_data)

            return [
                self.transform_model_to_openwebui_format(m) for m in model_data
            ]
        except requests.exceptions.RequestException as e:
            return [{"id": "error", "name": f"Error fetching models: {str(e)}"}]
        except Exception as e:
            return [{"id": "error", "name": f"Error fetching models: {str(e)}"}]

    def chat_completion(
        self,
        body: Dict[str, Any],
        *,
        stream: bool = False,
        preset: str = "",
        web_search: bool = False,
    ) -> Union[Dict[str, Any], Iterator[bytes]]:
        """
        Sends a chat completion request to OpenRouter.

        Args:
            body: Request body (model, messages, etc.).
            stream: Whether to stream the response.
            preset: Optional OpenRouter preset string.
            web_search: If True, append :online to the model ID.

        Returns:
            For stream=False: JSON response dict.
            For stream=True: Iterator of response bytes (raw lines).
        """
        if not self.config.api_key:
            return {
                "error": {
                    "code": "missing_api_key",
                    "message": "OpenRouter API Key not set in Valves.",
                }
            }

        headers = self.get_request_headers()
        model_id = self.extract_model_id(body["model"])
        if web_search:
            model_id += ":online"

        payload = {**body, "model": model_id}
        if preset:
            payload["preset"] = preset

        try:
            response = requests.post(
                url=f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                stream=stream,
            )

            if not response.ok:
                return self.handle_api_error(response)

            if stream:
                return response.iter_lines()
            return response.json()

        except requests.exceptions.RequestException as e:
            return self.handle_request_exception(e)
        except Exception as e:
            return self.handle_request_exception(e)
