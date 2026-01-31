"""
title: openrouter
author(s): jararered + gemini-3-pro
author_url(s): https://github.com/jararered/owui-openrouter + https://deepmind.google/models/gemini/pro/
version: 0.1.0
"""

from pydantic import BaseModel, Field
import requests
from typing import List, Union, Iterator, Optional
from decimal import Decimal, ROUND_HALF_UP


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
            description="Your OpenRouter API Key. Get one at openrouter.ai/keys",
        )
        OPENROUTER_PRESET: str = Field(
            default="",
            description="Optional: OpenRouter preset string (e.g., '@preset/lightning'). Set up presets at openrouter.ai.",
        )
        OPENROUTER_WEB_SEARCH: bool = Field(
            default=False,
            description="Optional: Whether to use OpenRouter's websearch functionality for all models. This comes at an additional cost listed on OpenRouter.",
        )
        SHOW_OPENROUTER_MODEL_PRICING: bool = Field(
            default=False,
            description="Optional: Whether to show pricing information for the models.",
        )
        STRIP_OPENROUTER_STREAM_COMMENTS: bool = Field(
            default=True,
            description="Optional: Whether to filter out stream comments (like ': OPENROUTER PROCESSING') from the response.",
        )
        MODEL_AUTHOR_WHITELIST: str = Field(
            default="",
            description="Optional: Comma-separated list of model authors to whitelist. (e.g. anthropic,google,openai,mistralai,meta-llama,x-ai)",
        )
        MODEL_AUTHOR_BLACKLIST: str = Field(
            default="",
            description="Optional: Comma-separated list of model authors to blacklist. (e.g. anthropic,google,openai,mistralai,meta-llama,x-ai)",
        )
        NAME_PREFIX: str = Field(
            default="",
            description="Prefix to be added before model names (e.g., OpenRouter/google/gemma).",
        )
        APPLICATION_NAME: str = Field(
            default="OpenWebUI",
            description="Optional: Your site name for OpenRouter rankings (X-Title header).",
        )
        APPLICATION_URL: str = Field(
            default="https://openwebui.com",
            description="Optional: Your site URL for OpenRouter rankings (HTTP-Referer header).",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.api_base = "https://openrouter.ai/api/v1"

    def _get_auth_headers(self) -> dict:
        """Creates authentication headers for API requests."""
        return {
            "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

    def _get_request_headers(self) -> dict:
        """Creates complete headers for chat completion requests."""
        headers = self._get_auth_headers()
        headers.update(
            {
                "HTTP-Referer": self.valves.APPLICATION_URL,
                "X-Title": self.valves.APPLICATION_NAME,
            }
        )
        return headers

    def _extract_model_id(self, model_string: str) -> str:
        """
        Extracts the actual OpenRouter model ID from the model string.

        The model ID usually comes in as "function_file_name.model_id"
        Example: "openrouter_pipe.google/gemma-7b" -> "google/gemma-7b"
        """
        if "." in model_string:
            return model_string[model_string.find(".") + 1 :]
        return model_string

    def _apply_author_whitelist(self, models: List[dict]) -> List[dict]:
        """Filters models by the specified authors."""
        if not self.valves.MODEL_AUTHOR_WHITELIST:
            return models

        authors = [author.strip() for author in self.valves.MODEL_AUTHOR_WHITELIST.split(",")]
        return [model for model in models if model["id"].split("/")[0] in authors]

    def _apply_author_blacklist(self, models: List[dict]) -> List[dict]:
        """Filters models by the specified authors."""
        if not self.valves.MODEL_AUTHOR_BLACKLIST:
            return models

        authors = [author.strip() for author in self.valves.MODEL_AUTHOR_BLACKLIST.split(",")]
        return [model for model in models if model["id"].split("/")[0] not in authors]

    def _format_pricing_string(self, prompt_price: str, completion_price: str) -> str:
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
        # Use Decimal for precise decimal arithmetic to avoid floating point precision issues
        price_per_million = Decimal(str(price_per_token)) * Decimal("1000000")

        # Round to 3 decimal places using banker's rounding (rounds 0.5 up)
        rounded = price_per_million.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # If it's a whole number, display as integer
        if rounded == rounded.to_integral_value():
            return str(int(rounded))

        # Otherwise, display with up to 2 decimal places, removing trailing zeros
        formatted = str(rounded)
        return formatted.rstrip("0").rstrip(".")

    def _format_model_pricing(self, model: dict) -> str:
        """Formats pricing information for a model."""
        prompt_price = self.format_price(float(model["pricing"]["prompt"]))
        completion_price = self.format_price(float(model["pricing"]["completion"]))
        return self._format_pricing_string(prompt_price, completion_price)

    def _transform_model_to_openwebui_format(self, model: dict) -> dict:
        """Transforms an OpenRouter model to OpenWebUI format."""
        pricing_display = (
            self._format_model_pricing(model)
            if self.valves.SHOW_OPENROUTER_MODEL_PRICING
            else ""
        )
        name = f"{self.valves.NAME_PREFIX}{model['id']}"
        if pricing_display:
            name = f"{name} ({pricing_display})"

        return {
            "id": model["id"],
            "name": name,
        }

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

    def _handle_api_error(self, response: requests.Response) -> dict:
        """Handles API error responses from OpenRouter."""
        try:
            error_data = response.json()
            return error_data
        except (ValueError, requests.exceptions.JSONDecodeError):
            # If response isn't JSON, return generic error
            return {
                "error": {
                    "code": response.status_code,
                    "message": f"HTTP {response.status_code}: {response.reason}",
                }
            }

    def _handle_request_exception(self, exception: Exception) -> dict:
        """Handles request exceptions with appropriate error formatting."""
        if isinstance(exception, requests.exceptions.RequestException):
            return {"error": {"code": "request_error", "message": str(exception)}}
        return {"error": {"code": "unknown_error", "message": str(exception)}}

    def pipes(self) -> List[dict]:
        """
        Fetches the list of available models from OpenRouter.
        Returns a list of model dictionaries in OpenWebUI format.
        """
        if not self.valves.OPENROUTER_API_KEY:
            return [ErrorModel("Error: OpenRouter API Key not set in Valves.")]

        try:
            headers = self._get_auth_headers()
            response = requests.get(f"{self.api_base}/models", headers=headers)
            response.raise_for_status()

            model_data = response.json()["data"]

            # Sort models by id alphabetically
            model_data.sort(key=lambda model: model["id"])

            # Apply author whitelist and blacklist
            model_data = self._apply_author_whitelist(model_data)
            model_data = self._apply_author_blacklist(model_data)

            # Transform to OpenWebUI format
            return [
                self._transform_model_to_openwebui_format(model) for model in model_data
            ]

        except requests.exceptions.RequestException as e:
            return [ErrorModel(f"Error fetching models: {str(e)}")]
        except Exception as e:
            return [ErrorModel(f"Error fetching models: {str(e)}")]

    def pipe(self, body: dict, __user__: dict) -> Union[str, Iterator[bytes], dict]:
        """
        Handles the chat completion request to OpenRouter.

        This is the main pipe function that processes requests and returns
        either streaming responses or complete JSON responses.
        """
        print(f"Pipe called for model: {body.get('model')}")

        if not self.valves.OPENROUTER_API_KEY:
            return {
                "error": {
                    "code": "missing_api_key",
                    "message": "OpenRouter API Key not set in Valves.",
                }
            }

        headers = self._get_request_headers()
        model_id = self._extract_model_id(body["model"])

        if self.valves.OPENROUTER_WEB_SEARCH:
            model_id += ":online"

        payload = {**body, "model": model_id}

        # Add preset if configured
        if self.valves.OPENROUTER_PRESET:
            payload["preset"] = self.valves.OPENROUTER_PRESET

        try:
            response = requests.post(
                url=f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers,
                stream=body.get("stream", False),
            )

            # Handle pre-stream errors (errors before any tokens are sent)
            if not response.ok:
                return self._handle_api_error(response)

            if body.get("stream", False):
                # Return iterator for streaming with filtered lines
                return (self.filter_stream_line(line) for line in response.iter_lines())
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            return self._handle_request_exception(e)
        except Exception as e:
            return self._handle_request_exception(e)
