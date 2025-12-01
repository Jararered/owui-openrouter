"""OpenRouter Integration for OpenWebUI.

This module provides integration with OpenRouter API for OpenWebUI, enabling access
to multiple AI models through a unified interface. It supports model filtering,
streaming responses, citation handling, and optional features like reasoning tokens
and cache control.

Attributes:
    __version__: Module version string (0.4.1)

Module Information:
    - Title: OpenRouter Integration for OpenWebUI
    - Version: 0.4.1
    - Author: kevarch
    - Author URL: https://github.com/kevarch
    - Contributor: Eloi Marques da Silva (https://github.com/eloimarquessilva)
    - Credits: rburmorrison (https://github.com/rburmorrison), Google Gemini Pro 2.5
    - License: MIT

Changelog:
    Version 0.4.1:
        - Contribution by Eloi Marques da Silva
        - Added FREE_ONLY parameter to optionally filter and display only free models
        - Changed MODEL_PREFIX and MODEL_PROVIDERS from required (str) to optional
          (Optional[str]), allowing null values
"""

import re
import requests
import json
import traceback  # Import traceback for detailed error logging
from typing import Optional, List, Union, Generator, Iterator, Callable
from pydantic import BaseModel, Field


def _insert_citations(text: str, citations: list[str]) -> str:
    """Replace citation markers in text with markdown links to citation URLs.

    This function processes text containing citation markers (e.g., [1], [2]) and
    replaces them with markdown-formatted links to the corresponding URLs from the
    citations list. The citation number in brackets corresponds to the index in the
    citations list (1-indexed in text, 0-indexed in list).

    Args:
        text: The text containing citation markers like [1], [2], etc.
        citations: A list of citation URLs, where index 0 corresponds to [1] in
            the text.

    Returns:
        str: Text with citation markers replaced with markdown links. Returns the
            original text if citations are empty or if an error occurs during
            processing.

    Example:
        >>> _insert_citations("See [1] for details", ["https://example.com"])
        'See [[1]](https://example.com) for details'
    """
    if not citations or not text:
        return text

    pattern = r"\[(\d+)\]"

    def replace_citation(match_obj):
        try:
            num = int(match_obj.group(1))
            if 1 <= num <= len(citations):
                url = citations[num - 1]
                return f"[[{num}]]({url})"
            else:
                return match_obj.group(0)
        except (ValueError, IndexError):
            return match_obj.group(0)

    try:
        return re.sub(pattern, replace_citation, text)
    except Exception as e:
        print(f"Error during citation insertion: {e}")
        return text


def _format_citation_list(citations: list[str]) -> str:
    """Format a list of citation URLs into a markdown-formatted string.

    This function takes a list of citation URLs and formats them as a numbered
    markdown list with a separator header. The output is suitable for appending
    to the end of a response.

    Args:
        citations: A list of citation URLs to format.

    Returns:
        str: A formatted markdown string with a separator and numbered list of
            citations (e.g., "\\n\\n---\\nCitations:\\n1. url1\\n2. url2").
            Returns an empty string if no citations are provided or if an error
            occurs.

    Example:
        >>> _format_citation_list(["https://example.com", "https://test.com"])
        '\\n\\n---\\nCitations:\\n1. https://example.com\\n2. https://test.com'
    """
    if not citations:
        return ""

    try:
        citation_list = [f"{i+1}. {url}" for i, url in enumerate(citations)]
        return "\n\n---\nCitations:\n" + "\n".join(citation_list)
    except Exception as e:
        print(f"Error formatting citation list: {e}")
        return ""

def _get_provider_logo(provider: str) -> str:
    """Retrieve the logo URL for a given model provider.

    This function maps provider names to their corresponding logo URLs. It supports
    common providers like OpenAI, Anthropic, Google, Meta, and Mistral AI. For
    unmapped providers, it returns a default placeholder logo.

    Args:
        provider: The provider name (case-insensitive). Typically extracted from
            model_id by splitting on "/" and taking the first part.

    Returns:
        str: The logo URL for the provider. Returns a default placeholder logo URL
            if the provider is not in the mapping.

    Example:
        >>> _get_provider_logo("openai")
        'https://assets.streamlinehq.com/image/...'
        >>> _get_provider_logo("unknown")
        'https://via.placeholder.com/50x50/cccccc/000000?text=AI'
    """
    provider_logos = {
        "openai": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/openai-wx0xqojo8lrv572wcvlcb.png/openai-twkvg10vdyltj9fklcgusg.png",
        "anthropic": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/1/anthropic-icon-wii9u8ifrjrd99btrqfgi.png/anthropic-icon-tdvkiqisswbrmtkiygb0ia.png",
        "google": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/3/google-icon-x87417ck9qy0ewcfgiv8.png/google-icon-yzx71jsaunfrqgy93lkp0c.png",
        "meta": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/4/meta-icon-w3lbidoopysan5m03f7159.png/meta-icon-totu193thz4r6sryadyus.png",
        "mistralai": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/4/mistral-ai-icon-3djkpjyks645ah3bg6zbxo.png/mistral-ai-icon-72wf09t6yllwqfky4jm3ql.png",
    }

    # Use a default logo for unmapped providers (e.g., a generic AI icon)
    default_logo = "https://via.placeholder.com/50x50/cccccc/000000?text=AI"
    return provider_logos.get(provider.lower(), default_logo)


class Pipe:
    """Main pipe class for OpenRouter integration with OpenWebUI.

    This class implements the OpenWebUI pipe interface, providing access to multiple
    AI models through the OpenRouter API. It handles model discovery, request
    processing, streaming responses, and various configuration options.

    Attributes:
        type: Always set to "manifold" to indicate this pipe provides multiple models.
        valves: Configuration object containing all user-configurable settings.
    """

    class Valves(BaseModel):
        """Configuration settings for the OpenRouter pipe.

        This Pydantic model defines all configurable parameters that users can set
        in OpenWebUI to customize the behavior of the OpenRouter integration.

        Attributes:
            OPENROUTER_API_KEY: OpenRouter API key (required for operation).
            INCLUDE_REASONING: Whether to request reasoning tokens from models.
            MODEL_PREFIX: Optional prefix to add to model names in OpenWebUI.
            REQUEST_TIMEOUT: Timeout for API requests in seconds (must be > 0).
            MODEL_PROVIDERS: Comma-separated list of providers to include/exclude.
            INVERT_PROVIDER_LIST: If True, MODEL_PROVIDERS becomes an exclude list.
            ENABLE_CACHE_CONTROL: Enable OpenRouter prompt caching for cost savings.
            FREE_ONLY: If True, only show free models in the model list.
        """
        OPENROUTER_API_KEY: str = Field(
            default="",
            description="Your OpenRouter API key (required).",
        )
        INCLUDE_REASONING: bool = Field(
            default=True,
            description="Request reasoning tokens from models that support it.",
        )
        MODEL_PREFIX: Optional[str] = Field(
            default=None,
            description="Optional prefix for model names in Open WebUI (e.g., 'OR: ').",
        )
        REQUEST_TIMEOUT: int = Field(
            default=90,
            description="Timeout for API requests in seconds.",
            gt=0,
        )
        MODEL_PROVIDERS: Optional[str] = Field(
            default=None,
            description="Comma-separated list of model providers to include or exclude. Leave empty to include all providers.",
        )
        INVERT_PROVIDER_LIST: bool = Field(
            default=False,
            description="If true, the above 'Model Providers' list becomes an *exclude* list instead of an *include* list.",
        )
        ENABLE_CACHE_CONTROL: bool = Field(
            default=False,
            description="Enable OpenRouter prompt caching by adding 'cache_control' to potentially large message parts. May reduce costs for supported models (e.g., Anthropic, Gemini) on subsequent calls with the same cached prefix. See OpenRouter docs for details.",
        )
        FREE_ONLY: bool = Field(
            default=False,
            description="If true, only free models will be available.",
        )

    def __init__(self) -> None:
        """Initialize the Pipe instance.

        Sets up the pipe type and configuration valves. Prints a warning if the
        OpenRouter API key is not configured.
        """
        self.type = "manifold"  # Specifies this pipe provides multiple models
        self.valves = self.Valves()
        if not self.valves.OPENROUTER_API_KEY:
            print("Warning: OPENROUTER_API_KEY is not set in Valves.")

    def pipes(self) -> List[dict]:
        """Fetch available models from the OpenRouter API.

        This method is called by OpenWebUI to discover the models this pipe provides.
        It retrieves the list of available models from OpenRouter, applies filtering
        based on provider settings and free-only option, and formats them for
        display in OpenWebUI.

        Returns:
            List[dict]: A list of dictionaries, each containing:
                - "id": The model identifier from OpenRouter
                - "name": The display name (with optional prefix) for OpenWebUI

            If an error occurs or no models are found, returns a list with a single
            error dictionary containing:
                - "id": "error"
                - "name": Error message describing what went wrong

        Raises:
            requests.exceptions.Timeout: If the API request times out.
            requests.exceptions.HTTPError: If the API returns an HTTP error.
            requests.exceptions.RequestException: If a network error occurs.
        """
        if not self.valves.OPENROUTER_API_KEY:
            return [
                {"id": "error", "name": "Pipe Error: OpenRouter API Key not provided"}
            ]

        try:
            headers = {"Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}"}
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            models_data = response.json()
            raw_models_data = models_data.get("data", [])
            models: List[dict] = []

            # --- Provider Filtering Logic ---
            provider_list_str = (self.valves.MODEL_PROVIDERS or "").lower()
            invert_list = self.valves.INVERT_PROVIDER_LIST
            target_providers = {
                p.strip() for p in provider_list_str.split(",") if p.strip()
            }
            # --- End Filtering Logic ---

            for model in raw_models_data:
                model_id = model.get("id")
                if not model_id:
                    continue

                # Apply Provider Filtering
                if target_providers:
                    provider = (
                        model_id.split("/", 1)[0].lower()
                        if "/" in model_id
                        else model_id.lower()
                    )
                    provider_in_list = provider in target_providers
                    keep = (provider_in_list and not invert_list) or (
                        not provider_in_list and invert_list
                    )
                    if not keep:
                        continue

                # Apply Free Only Filtering
                if self.valves.FREE_ONLY and "free" not in model_id.lower():
                    continue

                # Get the model name and prefix
                model_name = model.get("name", model_id)
                prefix = self.valves.MODEL_PREFIX or ""

                # Add the model to the list
                models.append({"id": model_id, "name": f"{prefix}{model_name}"})

            if not models:
                if self.valves.FREE_ONLY:
                    return [{"id": "error", "name": "Pipe Error: No free models found"}]
                elif target_providers:
                    return [
                        {
                            "id": "error",
                            "name": "Pipe Error: No models found matching the provider filter",
                        }
                    ]
                else:
                    return [
                        {
                            "id": "error",
                            "name": "Pipe Error: No models found on OpenRouter",
                        }
                    ]

            return models

        except requests.exceptions.Timeout:
            print("Error fetching models: Request timed out.")
            return [{"id": "error", "name": "Pipe Error: Timeout fetching models"}]
        except requests.exceptions.HTTPError as e:
            error_msg = f"Pipe Error: HTTP {e.response.status_code} fetching models"
            try:
                error_detail = e.response.json().get("error", {}).get("message", "")
                if error_detail:
                    error_msg += f": {error_detail}"
            except json.JSONDecodeError:
                pass
            print(f"Error fetching models: {error_msg} (URL: {e.request.url})")
            return [{"id": "error", "name": error_msg}]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: Request failed: {e}")
            return [
                {
                    "id": "error",
                    "name": f"Pipe Error: Network error fetching models: {e}",
                }
            ]
        except Exception as e:
            print(f"Unexpected error fetching models: {e}")
            traceback.print_exc()
            return [{"id": "error", "name": f"Pipe Error: Unexpected error: {e}"}]

    def pipe(self, body: dict) -> Union[str, Generator[str, None, None], Iterator[str]]:
        """Process incoming chat requests from OpenWebUI.

        This is the main function called by OpenWebUI when a user interacts with a
        model provided by this pipe. It handles both streaming and non-streaming
        requests, applies cache control if enabled, and processes the response
        with citation handling.

        Args:
            body: The request body conforming to OpenAI chat completions format.
                Expected keys include:
                - "model": The model identifier
                - "messages": List of message objects
                - "stream": Boolean indicating if streaming is requested
                - "http_referer": Optional referer URL
                - "x_title": Optional title for the request

        Returns:
            Union[str, Generator[str, None, None], Iterator[str]]: 
                - For non-streaming: A string containing the complete response
                - For streaming: A generator that yields response chunks as strings
                - Error messages are returned as strings

        Note:
            If the model ID contains a dot (.), it will be split and only the part
            after the dot will be used as the actual model identifier.
        """
        if not self.valves.OPENROUTER_API_KEY:
            return "Pipe Error: OpenRouter API Key is not configured."

        try:
            payload = body.copy()
            if "model" in payload and payload["model"] and "." in payload["model"]:
                payload["model"] = payload["model"].split(".", 1)[1]

            # --- Apply Cache Control Logic ---
            if self.valves.ENABLE_CACHE_CONTROL and "messages" in payload:
                try:
                    cache_applied = False
                    messages = payload["messages"]

                    # 1. Try applying to System Message
                    for i, msg in enumerate(messages):
                        if msg.get("role") == "system" and isinstance(
                            msg.get("content"), list
                        ):
                            longest_index, max_len = -1, -1
                            for j, part in enumerate(msg["content"]):
                                if part.get("type") == "text":
                                    text_len = len(part.get("text", ""))
                                    if text_len > max_len:
                                        max_len, longest_index = text_len, j
                            if longest_index != -1:
                                msg["content"][longest_index]["cache_control"] = {
                                    "type": "ephemeral"
                                }
                                cache_applied = True
                                break

                    # 2. Fallback to Last User Message
                    if not cache_applied:
                        for msg in reversed(messages):
                            if msg.get("role") == "user" and isinstance(
                                msg.get("content"), list
                            ):
                                longest_index, max_len = -1, -1
                                for j, part in enumerate(msg["content"]):
                                    if part.get("type") == "text":
                                        text_len = len(part.get("text", ""))
                                        if text_len > max_len:
                                            max_len, longest_index = text_len, j
                                if longest_index != -1:
                                    msg["content"][longest_index]["cache_control"] = {
                                        "type": "ephemeral"
                                    }
                                    break
                except Exception as cache_err:
                    print(f"Warning: Error applying cache_control logic: {cache_err}")
                    traceback.print_exc()
            # --- End Cache Control Logic ---

            if self.valves.INCLUDE_REASONING:
                payload["include_reasoning"] = True

            headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": body.get("http_referer", "https://openwebui.com/"),
                "X-Title": body.get("x_title", "Open WebUI via Pipe"),
            }

            url = "https://openrouter.ai/api/v1/chat/completions"
            is_streaming = body.get("stream", False)

            if is_streaming:
                return self.stream_response(
                    url,
                    headers,
                    payload,
                    _insert_citations,
                    _format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )
            else:
                return self.non_stream_response(
                    url,
                    headers,
                    payload,
                    _insert_citations,
                    _format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )

        except Exception as e:
            print(f"Error preparing request in pipe method: {e}")
            traceback.print_exc()
            return f"Pipe Error: Failed to prepare request: {e}"

    def non_stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        citation_inserter: Callable[[str, list[str]], str],
        citation_formatter: Callable[[list[str]], str],
        timeout: int,
    ) -> str:
        """Handle non-streaming API requests to OpenRouter.

        Sends a POST request to the OpenRouter API and processes the complete
        response. Handles reasoning tokens, citations, and formats the final
        output appropriately.

        Args:
            url: The API endpoint URL for chat completions.
            headers: HTTP headers including authorization and content type.
            payload: The request payload containing model, messages, etc.
            citation_inserter: Function to insert citations into text.
            citation_formatter: Function to format citation lists.
            timeout: Request timeout in seconds.

        Returns:
            str: The formatted response text including:
                - Reasoning content (if present) wrapped in <think> tags
                - Main content with citations inserted
                - Formatted citation list at the end
                - Error message if the request fails

        Raises:
            requests.exceptions.Timeout: If the request exceeds the timeout.
            requests.exceptions.HTTPError: If the API returns an HTTP error.
        """
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()

            res = response.json()
            if not res.get("choices"):
                return ""

            choice = res["choices"][0]
            message = choice.get("message", {})
            citations = res.get("citations", [])

            content = message.get("content", "")
            reasoning = message.get("reasoning", "")

            content = citation_inserter(content, citations)
            reasoning = citation_inserter(reasoning, citations)
            citation_list = citation_formatter(citations)

            final = ""
            if reasoning:
                final += f"<think>\n{reasoning}\n</think>\n\n"
            if content:
                final += content
            if final:
                final += citation_list
            return final

        except requests.exceptions.Timeout:
            return f"Pipe Error: Request timed out ({timeout}s)"
        except requests.exceptions.HTTPError as e:
            error_msg = f"Pipe Error: API returned HTTP {e.response.status_code}"
            try:
                detail = e.response.json().get("error", {}).get("message", "")
                if detail:
                    error_msg += f": {detail}"
            except Exception:
                pass
            return error_msg
        except Exception as e:
            print(f"Unexpected error in non_stream_response: {e}")
            traceback.print_exc()
            return f"Pipe Error: Unexpected error processing response: {e}"

    def stream_response(
        self,
        url: str,
        headers: dict,
        payload: dict,
        citation_inserter: Callable[[str, list[str]], str],
        citation_formatter: Callable[[list[str]], str],
        timeout: int,
    ) -> Generator[str, None, None]:
        """Handle streaming API requests to OpenRouter.

        Sends a streaming POST request to the OpenRouter API and yields response
        chunks as they arrive. Handles reasoning tokens, content chunks, and
        citations in real-time. Ensures proper formatting with reasoning tags and
        appends citations at the end.

        Args:
            url: The API endpoint URL for chat completions.
            headers: HTTP headers including authorization and content type.
            payload: The request payload containing model, messages, etc.
            citation_inserter: Function to insert citations into text chunks.
            citation_formatter: Function to format citation lists.
            timeout: Request timeout in seconds.

        Yields:
            str: Response chunks as they arrive from the API, including:
                - Opening <think> tag when reasoning starts
                - Reasoning chunks with citations inserted
                - Closing </think> tag when switching to content
                - Content chunks with citations inserted
                - Formatted citation list at the end

        Note:
            The generator ensures that:
            - Reasoning blocks are properly opened and closed
            - Citations are inserted into each chunk as they arrive
            - The citation list is yielded exactly once at the end
            - Response is properly closed even if errors occur
        """
        response = None
        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=True
            )
            response.raise_for_status()
            yielded_think_start = False  # Track if we've started the <think> block
            closed_think = False  # Track if we've closed the <think> block
            latest_citations: List[str] = []  # List of citations from the latest chunk
            citation_list_yielded = (
                False  # Ensure citation list is only yielded once at the end
            )
            for line in response.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[len(b"data: ") :].decode("utf-8")
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "choices" in chunk:
                    choice = chunk["choices"][0]
                    citations = chunk.get("citations")
                    if citations is not None:
                        latest_citations = citations
                    delta = choice.get("delta", {})
                    reasoning = delta.get("reasoning", "")
                    content = delta.get("content", "")
                    # Handle reasoning chunks
                    if reasoning:
                        # Yield the <think> start tag if this is the first reasoning chunk
                        if not yielded_think_start:
                            yield "<think>\n"
                            yielded_think_start = True
                        # Yield the reasoning chunk immediately with citations inserted
                        yield citation_inserter(reasoning, latest_citations)
                    # Handle content chunks
                    if content:
                        # If thinking was open and we're switching to content, close it
                        if yielded_think_start and not closed_think:
                            yield "\n</think>\n\n"
                            closed_think = True
                        # Yield the content chunk immediately with citations inserted
                        yield citation_inserter(content, latest_citations)
            # Final cleanup: Close the think block if it was left open (e.g., reasoning-only response)
            if yielded_think_start and not closed_think:
                yield "\n</think>\n\n"
            # Yield the final citation list once at the end
            if not citation_list_yielded:
                yield citation_formatter(latest_citations)
                citation_list_yielded = True
        except requests.exceptions.Timeout:
            yield f"Pipe Error: Request timed out"
            # Yield final citations if any were collected before error
            if latest_citations and not citation_list_yielded:
                yield citation_formatter(latest_citations)
        except requests.exceptions.HTTPError as e:
            yield f"Pipe Error: API returned HTTP {e.response.status_code}"
            # Yield final citations if any were collected before error
            if latest_citations and not citation_list_yielded:
                yield citation_formatter(latest_citations)
        except Exception as e:
            yield f"Pipe Error: Unexpected error during streaming: {e}"
            # Yield final citations if any were collected before error
            if latest_citations and not citation_list_yielded:
                yield citation_formatter(latest_citations)
        finally:
            if response:
                response.close()
