"""OpenRouter Pipe for Open WebUI

This module provides integration with OpenRouter API for OpenWebUI, enabling access
to multiple AI models through a unified interface. It supports model filtering,
streaming responses, citation handling, and optional features like reasoning tokens
and cache control.

Attributes:
    __version__: Module version string (0.5)

Module Information:
    - Title: OpenRouter Pipe for Open WebUI
    - Version: 0.5
    - License: MIT
"""

import re
import requests
import json
import traceback

from typing import Optional, List, Union, Generator, Iterator, Callable
from pydantic import BaseModel, Field


def insert_citations(text_content: str, citations: list[str]) -> str:
    """Replace citation markers in text with markdown links to citation URLs.

    This function processes text containing citation markers (e.g., [1], [2]) and
    replaces them with markdown-formatted links to the corresponding URLs from the
    citations list. The citation number in brackets corresponds to the index in the
    citations list (1-indexed in text, 0-indexed in list).

    Args:
        text_content: The text containing citation markers like [1], [2], etc.
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
    # Early return if no citations or empty text to avoid unnecessary processing
    if not citations or not text_content:
        return text_content

    # Regex pattern to match citation markers: [1], [2], [123], etc.
    # Captures the number inside brackets for processing
    citation_marker_pattern = r"\[(\d+)\]"

    def replace_citation(citation_match_object):
        """Inner function to replace each citation marker with a markdown link.

        This function is called by re.sub() for each match found in the text.
        It converts the citation number from the match to an integer, validates
        it against the citations list, and creates a markdown link.
        """
        try:
            # Extract the citation number from the regex match (group 1)
            citation_number = int(citation_match_object.group(1))

            # Validate citation number: must be between 1 and length of citations list
            # Note: Citations are 1-indexed in text ([1], [2]) but 0-indexed in list
            if 1 <= citation_number <= len(citations):
                # Convert to 0-based index and get the corresponding URL
                citation_url = citations[citation_number - 1]
                # Return markdown link format: [[1]](url)
                return f"[[{citation_number}]]({citation_url})"
            else:
                # Citation number out of range - return original marker unchanged
                return citation_match_object.group(0)
        except (ValueError, IndexError):
            # Handle edge cases: invalid number format or index errors
            # Return original marker to preserve text integrity
            return citation_match_object.group(0)

    try:
        # Apply regex substitution to entire text, using replace_citation for each match
        return re.sub(citation_marker_pattern, replace_citation, text_content)
    except Exception as e:
        # Log error but return original text to prevent breaking the response
        print(f"Error during citation insertion: {e}")
        return text_content


def format_citation_list(citations: list[str]) -> str:
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
    # Early return if no citations to format
    if not citations:
        return ""

    try:
        # Create numbered list items: "1. url1", "2. url2", etc.
        # enumerate() provides 0-based index, so we add 1 for 1-based numbering
        formatted_citation_list = [
            f"{citation_index+1}. {citation_url}"
            for citation_index, citation_url in enumerate(citations)
        ]

        # Combine separator, header, and numbered list with newlines
        # Format: "\n\n---\nCitations:\n1. url1\n2. url2\n..."
        return "\n\n---\nCitations:\n" + "\n".join(formatted_citation_list)
    except Exception as e:
        # Log error and return empty string to prevent breaking the response
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
    # Mapping of provider names (lowercase) to their logo URLs
    # These are StreamlineHQ asset URLs for major AI providers
    provider_logos = {
        "openai": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/logos/openai-wx0xqojo8lrv572wcvlcb.png/openai-twkvg10vdyltj9fklcgusg.png",
        "anthropic": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/1/anthropic-icon-wii9u8ifrjrd99btrqfgi.png/anthropic-icon-tdvkiqisswbrmtkiygb0ia.png",
        "google": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/3/google-icon-x87417ck9qy0ewcfgiv8.png/google-icon-yzx71jsaunfrqgy93lkp0c.png",
        "meta": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/4/meta-icon-w3lbidoopysan5m03f7159.png/meta-icon-totu193thz4r6sryadyus.png",
        "mistralai": "https://assets.streamlinehq.com/image/private/w_300,h_300,ar_1/f_auto/v1/icons/4/mistral-ai-icon-3djkpjyks645ah3bg6zbxo.png/mistral-ai-icon-72wf09t6yllwqfky4jm3ql.png",
    }

    # Default placeholder logo for providers not in the mapping
    # This ensures all providers have a logo, even if not explicitly mapped
    default_logo = "https://via.placeholder.com/50x50/cccccc/000000?text=AI"

    # Look up provider in mapping (case-insensitive via .lower())
    # Returns mapped logo or default if provider not found
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
        # Set pipe type to "manifold" to indicate this pipe provides multiple models
        # OpenWebUI uses this to determine how to handle model discovery
        self.type = "manifold"

        # Initialize configuration valves with default or user-provided values
        # Valves are Pydantic models that validate and store configuration
        self.valves = self.Valves()

        # Warn if API key is missing - pipe will still initialize but won't function
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
        # Validate API key before making any requests
        if not self.valves.OPENROUTER_API_KEY:
            return [
                {"id": "error", "name": "Pipe Error: OpenRouter API Key not provided"}
            ]

        try:
            # Prepare authorization header with Bearer token
            request_headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}"
            }

            # Fetch available models from OpenRouter API
            # This endpoint returns all models available through OpenRouter
            api_response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=request_headers,
                timeout=self.valves.REQUEST_TIMEOUT,
            )
            # Raise exception for HTTP error status codes (4xx, 5xx)
            api_response.raise_for_status()

            # Parse JSON response and extract models array
            # OpenRouter returns: {"data": [model1, model2, ...]}
            models_response_data = api_response.json()
            unfiltered_models_list = models_response_data.get("data", [])
            filtered_models: List[dict] = []

            # Parse provider filter string: convert to lowercase, split by comma,
            # strip whitespace, and create a set for O(1) lookup
            provider_filter_string = (self.valves.MODEL_PROVIDERS or "").lower()
            should_invert_provider_list = self.valves.INVERT_PROVIDER_LIST
            target_providers = {
                provider_name.strip()
                for provider_name in provider_filter_string.split(",")
                if provider_name.strip()
            }
            # Empty set means no filtering (all providers included)

            # Process each model from the API response
            for model_info in unfiltered_models_list:
                # Extract model ID (required field)
                model_id = model_info.get("id")
                if not model_id:
                    # Skip models without IDs (shouldn't happen, but defensive)
                    continue

                # Apply Provider Filtering
                # Only filter if provider list is specified (non-empty set)
                if target_providers:
                    # Extract provider name from model_id
                    # Format is typically "provider/model-name" or just "model-name"
                    # Split on "/" and take first part, or use entire ID if no "/"
                    model_provider = (
                        model_id.split("/", 1)[0].lower()
                        if "/" in model_id
                        else model_id.lower()
                    )
                    # Check if provider is in the target list
                    is_provider_in_filter_list = model_provider in target_providers

                    # Determine if model should be kept based on filter mode:
                    # - Include mode (not invert): keep if provider is in list
                    # - Exclude mode (invert): keep if provider is NOT in list
                    should_keep_model = (
                        is_provider_in_filter_list and not should_invert_provider_list
                    ) or (
                        not is_provider_in_filter_list and should_invert_provider_list
                    )
                    if not should_keep_model:
                        # Skip this model - doesn't match filter criteria
                        continue

                # Apply Free Only Filtering
                # If FREE_ONLY is enabled, only include models with "free" in their ID
                # OpenRouter marks free models with "free" in the model identifier
                if self.valves.FREE_ONLY and "free" not in model_id.lower():
                    continue

                # Get display name (use model's name field or fallback to ID)
                model_name = model_info.get("name", model_id)
                # Apply optional prefix (e.g., "OR: " to distinguish from other sources)
                model_name_prefix = self.valves.MODEL_PREFIX or ""

                # Add filtered model to results list
                # Format: {"id": "openrouter-model-id", "name": "prefix Display Name"}
                filtered_models.append(
                    {"id": model_id, "name": f"{model_name_prefix}{model_name}"}
                )

            # Handle case where no models passed the filters
            if not filtered_models:
                # Provide specific error messages based on which filter was applied
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
                    # No filters applied but still no models (unlikely but possible)
                    return [
                        {
                            "id": "error",
                            "name": "Pipe Error: No models found on OpenRouter",
                        }
                    ]

            # Return successfully filtered and formatted model list
            return filtered_models

        except requests.exceptions.Timeout:
            # Request exceeded the configured timeout
            print("Error fetching models: Request timed out.")
            return [{"id": "error", "name": "Pipe Error: Timeout fetching models"}]
        except requests.exceptions.HTTPError as http_error:
            # HTTP error response (4xx, 5xx status codes)
            error_message = (
                f"Pipe Error: HTTP {http_error.response.status_code} fetching models"
            )
            try:
                # Try to extract detailed error message from API response
                # OpenRouter may provide error details in JSON format
                error_detail = (
                    http_error.response.json().get("error", {}).get("message", "")
                )
                if error_detail:
                    error_message += f": {error_detail}"
            except json.JSONDecodeError:
                # Response is not valid JSON, use generic error message
                pass
            print(
                f"Error fetching models: {error_message} (URL: {http_error.request.url})"
            )
            return [{"id": "error", "name": error_message}]
        except requests.exceptions.RequestException as request_exception:
            # Network-level errors (connection refused, DNS failure, etc.)
            print(f"Error fetching models: Request failed: {request_exception}")
            return [
                {
                    "id": "error",
                    "name": f"Pipe Error: Network error fetching models: {request_exception}",
                }
            ]
        except Exception as unexpected_error:
            # Catch-all for any unexpected errors (programming errors, etc.)
            print(f"Unexpected error fetching models: {unexpected_error}")
            traceback.print_exc()
            return [
                {
                    "id": "error",
                    "name": f"Pipe Error: Unexpected error: {unexpected_error}",
                }
            ]

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
        # Validate API key before processing request
        if not self.valves.OPENROUTER_API_KEY:
            return "Pipe Error: OpenRouter API Key is not configured."

        try:
            # Create a copy of the request body to avoid modifying the original
            request_payload = body.copy()

            # Handle model ID format: OpenWebUI may prefix model IDs with a dot notation
            # (e.g., "pipe.model-name") - we need to extract just the model name
            if (
                "model" in request_payload
                and request_payload["model"]
                and "." in request_payload["model"]
            ):
                # Split on first dot and take everything after it
                # Example: "pipe.openai/gpt-4" -> "openai/gpt-4"
                request_payload["model"] = request_payload["model"].split(".", 1)[1]

            # OpenRouter supports prompt caching to reduce costs on repeated prefixes
            # We apply cache_control to the longest text part in system/user messages
            if self.valves.ENABLE_CACHE_CONTROL and "messages" in request_payload:
                try:
                    cache_control_applied = False
                    message_list = request_payload["messages"]

                    # Strategy 1: Try applying cache_control to System Message
                    # System messages are typically the longest and most stable parts
                    for message_index, message in enumerate(message_list):
                        # Only process system messages with list-based content
                        if message.get("role") == "system" and isinstance(
                            message.get("content"), list
                        ):
                            # Find the longest text part in the message
                            longest_part_index, maximum_text_length = -1, -1
                            for part_index, content_part in enumerate(
                                message["content"]
                            ):
                                if content_part.get("type") == "text":
                                    text_content_length = len(
                                        content_part.get("text", "")
                                    )
                                    if text_content_length > maximum_text_length:
                                        maximum_text_length, longest_part_index = (
                                            text_content_length,
                                            part_index,
                                        )

                            # Apply ephemeral cache control to longest text part
                            # "ephemeral" means cache is cleared after conversation ends
                            if longest_part_index != -1:
                                message["content"][longest_part_index][
                                    "cache_control"
                                ] = {"type": "ephemeral"}
                                cache_control_applied = True
                                break

                    # Strategy 2: Fallback to Last User Message
                    # If no system message found, use the most recent user message
                    if not cache_control_applied:
                        # Iterate backwards to find the last user message
                        for message in reversed(message_list):
                            if message.get("role") == "user" and isinstance(
                                message.get("content"), list
                            ):
                                # Find longest text part in user message
                                longest_part_index, maximum_text_length = -1, -1
                                for part_index, content_part in enumerate(
                                    message["content"]
                                ):
                                    if content_part.get("type") == "text":
                                        text_content_length = len(
                                            content_part.get("text", "")
                                        )
                                        if text_content_length > maximum_text_length:
                                            maximum_text_length, longest_part_index = (
                                                text_content_length,
                                                part_index,
                                            )

                                # Apply cache control to longest part
                                if longest_part_index != -1:
                                    message["content"][longest_part_index][
                                        "cache_control"
                                    ] = {"type": "ephemeral"}
                                    break
                except Exception as cache_control_error:
                    # Log cache control errors but don't fail the request
                    # Cache control is an optimization, not a requirement
                    print(
                        f"Warning: Error applying cache_control logic: {cache_control_error}"
                    )
                    traceback.print_exc()

            # Add reasoning token request if enabled
            # Some models (like o1) support reasoning tokens that show internal thinking
            if self.valves.INCLUDE_REASONING:
                request_payload["include_reasoning"] = True

            # Prepare HTTP headers for OpenRouter API request
            request_headers = {
                "Authorization": f"Bearer {self.valves.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                # OpenRouter uses these headers for analytics and attribution
                "HTTP-Referer": body.get("http_referer", "https://openwebui.com/"),
                "X-Title": body.get("x_title", "Open WebUI via Pipe"),
            }

            # OpenRouter chat completions endpoint
            openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"

            # Determine if client requested streaming response
            is_streaming_request = body.get("stream", False)

            # Route to appropriate handler based on streaming preference
            if is_streaming_request:
                # Streaming: yield chunks as they arrive for real-time display
                return self.stream_response(
                    openrouter_api_url,
                    request_headers,
                    request_payload,
                    insert_citations,
                    format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )
            else:
                # Non-streaming: wait for complete response and return all at once
                return self.non_stream_response(
                    openrouter_api_url,
                    request_headers,
                    request_payload,
                    insert_citations,
                    format_citation_list,
                    self.valves.REQUEST_TIMEOUT,
                )

        except Exception as request_preparation_error:
            print(
                f"Error preparing request in pipe method: {request_preparation_error}"
            )
            traceback.print_exc()
            return f"Pipe Error: Failed to prepare request: {request_preparation_error}"

    def non_stream_response(
        self,
        api_endpoint_url: str,
        request_headers: dict,
        request_payload: dict,
        citation_inserter: Callable[[str, list[str]], str],
        citation_formatter: Callable[[list[str]], str],
        request_timeout: int,
    ) -> str:
        """Handle non-streaming API requests to OpenRouter.

        Sends a POST request to the OpenRouter API and processes the complete
        response. Handles reasoning tokens, citations, and formats the final
        output appropriately.

        Args:
            api_endpoint_url: The API endpoint URL for chat completions.
            request_headers: HTTP headers including authorization and content type.
            request_payload: The request payload containing model, messages, etc.
            citation_inserter: Function to insert citations into text.
            citation_formatter: Function to format citation lists.
            request_timeout: Request timeout in seconds.

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
            # Send POST request to OpenRouter API with JSON payload
            api_response = requests.post(
                api_endpoint_url,
                headers=request_headers,
                json=request_payload,
                timeout=request_timeout,
            )
            # Raise exception for HTTP error status codes
            api_response.raise_for_status()

            # Parse JSON response from API
            response_data = api_response.json()

            # Validate response structure - must have choices array
            if not response_data.get("choices"):
                # Empty response (shouldn't happen normally)
                return ""

            # Extract first choice (OpenRouter typically returns one choice)
            first_choice = response_data["choices"][0]
            response_message = first_choice.get("message", {})

            # Extract citations if present (some models provide source citations)
            citation_urls = response_data.get("citations", [])

            # Extract content and reasoning from message
            # Reasoning tokens show internal model thinking (o1, etc.)
            response_content = response_message.get("content", "")
            response_reasoning = response_message.get("reasoning", "")

            # Insert citation markers into both content and reasoning text
            # This converts [1], [2] markers into clickable markdown links
            response_content = citation_inserter(response_content, citation_urls)
            response_reasoning = citation_inserter(response_reasoning, citation_urls)

            # Format citations as a numbered list for appending to response
            formatted_citation_list = citation_formatter(citation_urls)

            # Build final response string with proper formatting
            final_response_text = ""

            # Add reasoning block if present (wrapped in special tags for UI)
            # OpenWebUI uses <think> tags to hide/show reasoning
            if response_reasoning:
                final_response_text += f"<think>\n{response_reasoning}\n</think>\n\n"

            # Add main content after reasoning
            if response_content:
                final_response_text += response_content

            # Append citation list at the end if we have any content
            if final_response_text:
                final_response_text += formatted_citation_list

            return final_response_text

        except requests.exceptions.Timeout:
            # Request exceeded timeout - return user-friendly error
            return f"Pipe Error: Request timed out ({request_timeout}s)"
        except requests.exceptions.HTTPError as http_error:
            # HTTP error response - try to extract detailed error message
            error_message = (
                f"Pipe Error: API returned HTTP {http_error.response.status_code}"
            )
            try:
                # OpenRouter may provide error details in JSON response
                error_detail = (
                    http_error.response.json().get("error", {}).get("message", "")
                )
                if error_detail:
                    error_message += f": {error_detail}"
            except Exception:
                # Response is not JSON or doesn't have expected structure
                pass
            return error_message
        except Exception as unexpected_error:
            # Catch-all for unexpected errors (parsing, logic errors, etc.)
            print(f"Unexpected error in non_stream_response: {unexpected_error}")
            traceback.print_exc()
            return (
                f"Pipe Error: Unexpected error processing response: {unexpected_error}"
            )

    def stream_response(
        self,
        api_endpoint_url: str,
        request_headers: dict,
        request_payload: dict,
        citation_inserter: Callable[[str, list[str]], str],
        citation_formatter: Callable[[list[str]], str],
        request_timeout: int,
    ) -> Generator[str, None, None]:
        """Handle streaming API requests to OpenRouter.

        Sends a streaming POST request to the OpenRouter API and yields response
        chunks as they arrive. Handles reasoning tokens, content chunks, and
        citations in real-time. Ensures proper formatting with reasoning tags and
        appends citations at the end.

        Args:
            api_endpoint_url: The API endpoint URL for chat completions.
            request_headers: HTTP headers including authorization and content type.
            request_payload: The request payload containing model, messages, etc.
            citation_inserter: Function to insert citations into text chunks.
            citation_formatter: Function to format citation lists.
            request_timeout: Request timeout in seconds.

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
        # Initialize response variable for cleanup in finally block
        streaming_response = None
        try:
            # Send streaming POST request - stream=True enables chunked response handling
            streaming_response = requests.post(
                api_endpoint_url,
                headers=request_headers,
                json=request_payload,
                stream=True,
            )
            # Raise exception for HTTP error status codes
            streaming_response.raise_for_status()

            # State tracking for streaming response formatting
            has_yielded_reasoning_start_tag = (
                False  # Track if we've started the <think> block
            )
            has_closed_reasoning_block = (
                False  # Track if we've closed the <think> block
            )
            latest_citation_urls: List[str] = (
                []
            )  # List of citations from the latest chunk
            has_yielded_citation_list = (
                False  # Ensure citation list is only yielded once at the end
            )

            # Process streaming response line by line
            # OpenRouter uses Server-Sent Events (SSE) format: "data: {...}\n\n"
            for response_line in streaming_response.iter_lines():
                # Skip empty lines and non-data lines (comments, etc.)
                if not response_line or not response_line.startswith(b"data: "):
                    continue

                # Extract JSON data from SSE line format: "data: {...}"
                sse_data_string = response_line[len(b"data: ") :].decode("utf-8")

                # OpenRouter sends "[DONE]" to signal end of stream
                if sse_data_string == "[DONE]":
                    break

                # Parse JSON chunk from SSE data line
                try:
                    response_chunk = json.loads(sse_data_string)
                except json.JSONDecodeError:
                    # Skip malformed JSON chunks (shouldn't happen but defensive)
                    continue

                # Process chunk if it contains choices array
                if "choices" in response_chunk:
                    first_choice = response_chunk["choices"][0]

                    # Update citations if present in this chunk
                    # Citations may be updated throughout the stream
                    chunk_citations = response_chunk.get("citations")
                    if chunk_citations is not None:
                        latest_citation_urls = chunk_citations

                    # Extract delta (incremental changes) from choice
                    # Delta contains only the new content since last chunk
                    content_delta = first_choice.get("delta", {})
                    reasoning_chunk = content_delta.get("reasoning", "")
                    content_chunk = content_delta.get("content", "")

                    # Handle reasoning chunks (internal model thinking)
                    if reasoning_chunk:
                        # Yield opening tag on first reasoning chunk
                        # This marks the start of reasoning content for UI rendering
                        if not has_yielded_reasoning_start_tag:
                            yield "<think>\n"
                            has_yielded_reasoning_start_tag = True

                        # Yield reasoning chunk with citations inserted in real-time
                        # Citations are inserted as they appear in the stream
                        yield citation_inserter(reasoning_chunk, latest_citation_urls)

                    # Handle content chunks (actual response text)
                    if content_chunk:
                        # Transition from reasoning to content: close reasoning block
                        # This happens when model switches from thinking to responding
                        if (
                            has_yielded_reasoning_start_tag
                            and not has_closed_reasoning_block
                        ):
                            yield "\n</think>\n\n"
                            has_closed_reasoning_block = True

                        # Yield content chunk with citations inserted in real-time
                        yield citation_inserter(content_chunk, latest_citation_urls)

            # Final cleanup: Close reasoning block if it was left open
            # This handles edge case where response is reasoning-only (no content)
            if has_yielded_reasoning_start_tag and not has_closed_reasoning_block:
                yield "\n</think>\n\n"

            # Yield formatted citation list exactly once at the end of stream
            # This ensures citations appear after all content, not scattered throughout
            if not has_yielded_citation_list:
                yield citation_formatter(latest_citation_urls)
                has_yielded_citation_list = True
        except requests.exceptions.Timeout:
            # Request exceeded timeout - yield error message
            yield f"Pipe Error: Request timed out"
            # Yield final citations if any were collected before timeout
            # This ensures partial citations aren't lost on error
            if latest_citation_urls and not has_yielded_citation_list:
                yield citation_formatter(latest_citation_urls)
        except requests.exceptions.HTTPError as http_error:
            # HTTP error response - yield error with status code
            yield f"Pipe Error: API returned HTTP {http_error.response.status_code}"
            # Yield final citations if any were collected before error
            if latest_citation_urls and not has_yielded_citation_list:
                yield citation_formatter(latest_citation_urls)
        except Exception as streaming_error:
            # Catch-all for unexpected errors during streaming
            yield f"Pipe Error: Unexpected error during streaming: {streaming_error}"
            # Yield final citations if any were collected before error
            if latest_citation_urls and not has_yielded_citation_list:
                yield citation_formatter(latest_citation_urls)
        finally:
            # Always close response connection to free resources
            # This is critical for streaming responses to prevent connection leaks
            if streaming_response:
                streaming_response.close()
