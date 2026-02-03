import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Type

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel





class VLMClient:
    """
    A client for interacting with Vision Language Models (VLMs) via OpenAI-compatible APIs.
    Supports both synchronous and asynchronous calls with automatic retries.

    Attributes:
        model_name: Name of the model to use.
        max_tokens: Maximum number of tokens for the completion.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        max_retries: Number of retry attempts.
        retry_delay: Delay between retries in seconds.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        timeout: int = 300,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize the VLMClient.

        Args:
            base_url: The base URL for the API.
            api_key: The API key. If not provided, it will look for the OPENAI_API_KEY environment variable.
            model_name: The name of the model to use.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
            retry_delay: Delay between retries in seconds.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Ensure we have an API key or a placeholder for local servers
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")

        self.client = OpenAI(base_url=base_url, api_key=self.api_key, timeout=timeout)

        self.async_client = AsyncOpenAI(base_url=base_url, api_key=self.api_key, timeout=timeout)

    def call(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """
        Make a synchronous call to the VLM.

        Args:
            image_b64: Base64 encoded image string.
            mime_type: The MIME type of the image.
            system_prompt: Optional system prompt override.
            user_prompt: Optional user prompt override.
            output_schema: Optional Pydantic model for structured output parsing.

        Returns:
            The API response object.

        Raises:
            RuntimeError: If the request fails after all retry attempts.
        """
        messages = self._build_message(
            image_b64, mime_type, system_prompt, user_prompt, output_schema
        )

        for attempt in range(self.max_retries):
            try:
                if output_schema:
                    return self.client.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        response_format=output_schema,
                    )
                else:
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"VLM call failed after {self.max_retries} attempts: {str(e)}"
                    ) from e

    async def acall(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """
        Make an asynchronous call to the VLM.

        Args:
            image_b64: Base64 encoded image string.
            mime_type: The MIME type of the image.
            system_prompt: Optional system prompt override.
            user_prompt: Optional user prompt override.
            output_schema: Optional Pydantic model for structured output parsing.

        Returns:
            The API response object.

        Raises:
            RuntimeError: If the request fails after all retry attempts.
        """
        messages = self._build_message(
            image_b64, mime_type, system_prompt, user_prompt, output_schema
        )

        for attempt in range(self.max_retries):
            try:
                if output_schema:
                    return await self.async_client.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        response_format=output_schema,
                    )
                else:
                    return await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    await asyncio.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Asynchronous VLM call failed after {self.max_retries} attempts: {str(e)}"
                    ) from e

    def _build_message(
        self,
        image_b64: str,
        mime_type: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Construct the message payload for the API call.

        Args:
            image_b64: Base64 encoded image string.
            mime_type: The MIME type of the image.
            system_prompt: Optional system prompt override.
            user_prompt: Optional user prompt override.
            output_schema: Optional output schema.

        Returns:
            A list of message dictionaries.
        """
        messages = []

        if output_schema is not None:
            if not system_prompt:
                raise ValueError(
                    "When using response_format, you MUST provide a system_prompt explicitly "
                    "(default XML prompt is disabled because it conflicts with structured output)."
                )
            final_system_prompt = system_prompt
        else:
            from ..workflows import DEFAULT_SYSTEM_PROMPT, TRANSCRIPTION

            if system_prompt:
                final_system_prompt = f"{system_prompt}\n\n{TRANSCRIPTION}"
            else:
                final_system_prompt = DEFAULT_SYSTEM_PROMPT

        messages.append({"role": "system", "content": final_system_prompt})

        from ..workflows import DEFAULT_USER_PROMPT

        user_content = [
            {"type": "text", "text": user_prompt or DEFAULT_USER_PROMPT},
            {

                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
            },
        ]

        messages.append({"role": "user", "content": user_content})
        return messages
