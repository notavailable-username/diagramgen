"""Main application for Singapore Math Bar Model Generation.

This module orchestrates the interaction between:
- Google Gemini LLM (gemini-2.5-flash)
- MCP servers providing bar model drawing tools
- Streaming chat interface

The application loads Singapore Math word problems and generates
bar model visualizations to help solve them.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import mcp
import mcp.types
from google import genai
from google.genai import errors, types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import settings, get_images_dir, get_mcp_servers_dir, get_project_root


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Singapore Math Bar Model Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Run with default settings
  python main.py --thinking                   # Enable thinking mode (hidden)
  python main.py --thinking --show-thinking   # Enable thinking with display
  python main.py --no-examples                # Skip example questions
  python main.py --thinking --log-level INFO  # Enable thinking with info logging

Interactive Commands (toggles):
  /thinking      - Toggle thinking mode on/off (recreates chat session)
  /show-thinking - Toggle thinking display on/off (recreates chat session)
  /help          - Show help message
  /quit or /exit - Exit the application
        """
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking mode to see LLM's reasoning process"
    )

    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Display thinking output (only effective when --thinking is enabled)"
    )

    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip example questions and go directly to interactive mode"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=settings.LOG_LEVEL,
        help="Set logging level (default: %(default)s)"
    )

    return parser.parse_args()


class Server:
    """Manages MCP server connections and tool execution.

    This class handles the lifecycle of an MCP server connection,
    including initialization, tool/prompt listing, and cleanup.
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """Initialize an MCP server connection manager.

        Args:
            name: Name identifier for the server.
            config: Server configuration containing 'command' and 'args'.
        """
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection.

        Raises:
            Exception: If initialization fails.
        """
        server_params = StdioServerParameters(
            command=self.config["command"],
            args=self.config["args"],
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            logger.info(f"Server '{self.name}' initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_prompts(self) -> list[Any]:
        """List available prompts from the server.

        Returns:
            A list of available prompts.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        prompts_response = await self.session.list_prompts()
        prompts = []
        for item in prompts_response:
            if isinstance(item, tuple) and item[0] == "prompts":
                prompts.extend(
                    Prompt(prompt.name, prompt.description, prompt.arguments)
                    for prompt in item[1]
                )

        return prompts

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []
        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def get_prompt(self, prompt_name: str) -> Any:
        """Get a specific prompt by name.

        Args:
            prompt_name: Name of the prompt to retrieve.

        Returns:
            The requested prompt.

        Raises:
            RuntimeError: If server is not initialized.
            ValueError: If the prompt is not found.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        return await self.session.get_prompt(prompt_name)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts (default: 2).
            delay: Delay between retries in seconds (default: 1.0).

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        last_exception = None
        while attempt < retries:
            try:
                logger.info(f"Executing tool '{tool_name}' (attempt {attempt + 1}/{retries})")
                logger.info(f"Tool arguments: {arguments}")
                result = await self.session.call_tool(tool_name, arguments)
                logger.info(f"Tool '{tool_name}' executed successfully")
                return result

            except Exception as e:
                attempt += 1
                last_exception = e
                logger.error(
                    f"Error executing tool '{tool_name}': {type(e).__name__}: {str(e)}. "
                    f"Attempt {attempt} of {retries}.",
                    exc_info=True
                )
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached. Failing.")
                    # Raise the last exception with full details
                    raise Exception(
                        f"Tool '{tool_name}' failed after {retries} attempts. "
                        f"Last error: {type(last_exception).__name__}: {str(last_exception)}"
                    ) from last_exception

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
                logger.info(f"Server '{self.name}' cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any]
    ) -> None:
        """Initialize a tool.

        Args:
            name: Tool name.
            description: Tool description.
            input_schema: JSON schema for tool input.
        """
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_gemini(self) -> dict[str, Any]:
        """Format tool information for Gemini.

        Returns:
            A dictionary describing the tool in Gemini's expected format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


class Prompt:
    """Represents a prompt with its properties."""

    def __init__(
        self,
        name: str,
        description: str,
        arguments: dict[str, Any]
    ) -> None:
        """Initialize a prompt.

        Args:
            name: Prompt name.
            description: Prompt description.
            arguments: Prompt arguments schema.
        """
        self.name: str = name
        self.description: str = description
        self.arguments: dict[str, Any] = arguments

    def format_for_gemini(self) -> dict[str, Any]:
        """Format prompt information for Gemini.

        Returns:
            A dictionary describing the prompt.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments,
        }


class LLMClient:
    """Manages communication with the Google Gemini LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int,
        time_delay: float,
        enable_thinking: bool = False,
        show_thinking: bool = True
    ) -> None:
        """Initialize LLM client.

        Args:
            api_key: Google Gemini API key.
            model: Model identifier (e.g., 'gemini-2.5-flash').
            max_retries: Maximum retry attempts for API calls.
            time_delay: Delay between retries in seconds.
            enable_thinking: Enable thinking mode in the LLM configuration.
            show_thinking: Display thinking output to user (only if enable_thinking=True).
        """
        self.api_key: str = api_key
        self.model: str = model
        self.max_retries: int = max_retries
        self.time_delay: float = time_delay
        self.enable_thinking: bool = enable_thinking
        self.show_thinking: bool = show_thinking
        self.client: Any = None
        self.chat: Any = None

    async def start_client_gemini(self) -> None:
        """Initialize the Gemini LLM async client.

        Uses the latest Google GenAI SDK pattern with .aio for async operations.
        """
        try:
            self.client = genai.Client(api_key=self.api_key).aio
            logger.info(f"Gemini async client initialized with model '{self.model}'")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    async def create_chat_gemini(self, config: types.GenerateContentConfig) -> None:
        """Create a chat session with the Gemini LLM.

        Args:
            config: Generation configuration including system instructions,
                   tools, and other settings.
        """
        try:
            self.chat = self.client.chats.create(
                model=self.model,
                config=config
            )
            thinking_status = "enabled" if self.enable_thinking else "disabled"
            logger.info(f"Gemini chat session created (thinking: {thinking_status})")
        except Exception as e:
            logger.error(f"Failed to create Gemini chat: {e}")
            raise

    def toggle_thinking_mode(self) -> bool:
        """Toggle thinking mode on/off.

        Returns:
            New state of thinking mode (True = enabled, False = disabled).
        """
        self.enable_thinking = not self.enable_thinking
        status = "enabled" if self.enable_thinking else "disabled"
        logger.info(f"Thinking mode {status}")
        return self.enable_thinking

    def toggle_show_thinking(self) -> bool:
        """Toggle thinking display on/off.

        Returns:
            New state of show_thinking (True = shown, False = hidden).
        """
        self.show_thinking = not self.show_thinking
        status = "shown" if self.show_thinking else "hidden"
        logger.info(f"Thinking display {status}")
        return self.show_thinking

    async def get_response_gemini(self, messages: Any) -> Any:
        """Get a streaming response from the Gemini LLM.

        Args:
            messages: Message content or list of messages to send.

        Returns:
            Async iterator of response chunks, or error message string.
        """
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                response = await self.chat.send_message_stream(messages)
                return response
            except Exception as e:
                logger.error(f"Error getting Gemini response: {e}")

                # Handle server errors (500, 503 UNAVAILABLE)
                if isinstance(e, errors.ServerError):
                    logger.error(
                        f'Server Error - Code: {e.code}, '
                        f'Message: {e.message}, Status: {e.status}'
                    )
                    if e.code in [500, 503]:
                        # 503 UNAVAILABLE: The model is overloaded
                        if e.code == 503:
                            logger.warning(
                                f"Model overloaded (503). "
                                f"Attempt {attempt}/{self.max_retries}. "
                                f"Retrying in {self.time_delay} seconds..."
                            )
                            print(
                                f"\n[Model is overloaded. Retrying in "
                                f"{self.time_delay} seconds... "
                                f"(attempt {attempt}/{self.max_retries})]"
                            )
                        else:
                            logger.info(
                                f"Server error (500). "
                                f"Attempt {attempt}/{self.max_retries}: {e.message}"
                            )
                        await asyncio.sleep(self.time_delay)
                        continue

                # Handle rate limiting (429)
                if isinstance(e, errors.APIError):
                    rate_limit_time_delay = 10
                    logger.error(
                        f'API Error - Code: {e.code}, '
                        f'Message: {e.message}, Status: {e.status}'
                    )
                    if e.code == 429:
                        logger.warning(
                            f"Rate limit exceeded. Retrying after "
                            f"{rate_limit_time_delay} seconds."
                        )
                        print(
                            f"\n[Rate limit exceeded. Retrying in "
                            f"{rate_limit_time_delay} seconds...]"
                        )
                        await asyncio.sleep(rate_limit_time_delay)
                        continue

                # For other errors, return error message
                return (
                    f"I've encountered an error: {e}. "
                    f"Please try again or rephrase your request."
                )

        return (
            "I was unable to get a response from the LLM after multiple attempts. "
            "Please try again later."
        )


class Chat:
    """Handles the chat session with the LLM and tools."""

    def __init__(self, llm_client: LLMClient, servers: list[Server]) -> None:
        """Initialize a chat session.

        Args:
            llm_client: The LLM client for communication.
            servers: List of MCP servers providing tools and prompts.
        """
        self.llm_client: LLMClient = llm_client
        self.messages: list[dict[str, str]] = []
        self.servers: list[Server] = servers
        self.all_tools: list[Tool] = []

    async def start(self) -> None:
        """Start a new chat by gathering tools and prompts from servers.

        Raises:
            RuntimeError: If initialization fails.
        """
        system_message: Optional[str] = None
        bar_model_message: Optional[str] = None

        # Gather tools from all servers
        for server in self.servers:
            try:
                tools = await server.list_tools()
                self.all_tools.extend(tools)
                logger.info(
                    f"Loaded {len(tools)} tools from server '{server.name}'"
                )
            except Exception as e:
                logger.error(f"Error listing tools from server {server.name}: {e}")
                raise RuntimeError(e)

            # Get prompts
            try:
                # Get system prompt
                system_message_obj = await server.get_prompt("system-prompt")
                system_message = system_message_obj.messages[0].content
                if isinstance(system_message, mcp.types.TextContent):
                    system_message = system_message.text

                # Get bar model prompt
                try:
                    bar_model_message_obj = await server.get_prompt("bar-model-drawing")
                    bar_model_message = bar_model_message_obj.messages[0].content
                    if isinstance(bar_model_message, mcp.types.TextContent):
                        bar_model_message = bar_model_message.text
                except Exception:
                    bar_model_message = None
            except Exception:
                continue

        # Combine system prompt and bar model prompt
        if system_message and bar_model_message:
            system_message = f"{system_message}\n\n{bar_model_message}"

        # Create chat with Gemini
        try:
            function_declarations_gemini = [
                tool.format_for_gemini() for tool in self.all_tools
            ]

            # Create configuration
            config_params: dict[str, Any] = {
                "system_instruction": system_message,
                "tools": [
                    types.Tool(
                        function_declarations=function_declarations_gemini
                    )
                ],
                "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            }

            # Configure thinking mode using Google Gemini's ThinkingConfig
            #
            # thinking_budget parameter:
            #   0  = Disable thinking completely (no internal reasoning)
            #   -1 = Enable dynamic thinking (optimal, LLM decides budget)
            #   >0 = Fixed thinking budget (specific token limit)
            #
            # include_thoughts parameter:
            #   True  = Show thinking output to user (in response stream)
            #   False = Hide thinking output (thinking happens but not displayed)
            #
            # Recommended combinations:
            #   - Off: thinking_budget=0
            #   - On (hidden): thinking_budget=-1, include_thoughts=False
            #   - On (visible): thinking_budget=-1, include_thoughts=True
            if self.llm_client.enable_thinking:
                try:
                    config_params["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=-1,  # Dynamic thinking
                        include_thoughts=self.llm_client.show_thinking
                    )
                    logger.info(
                        f"Thinking mode enabled in configuration "
                        f"(display: {self.llm_client.show_thinking})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not enable thinking mode: {e}. "
                        "Continuing without thinking display."
                    )
            else:
                # Explicitly disable thinking
                try:
                    config_params["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=0
                    )
                    logger.info("Thinking mode disabled in configuration")
                except Exception as e:
                    logger.warning(f"Could not set thinking config: {e}")

            config = types.GenerateContentConfig(**config_params)

            await self.llm_client.create_chat_gemini(config)
            logger.info("Chat session initialized with system prompts and tools")
        except Exception as e:
            logger.error(f"Failed to create chat session with LLM: {e}")
            raise RuntimeError(e)

    async def process_function_calls(
        self,
        function_calls: Any
    ) -> tuple[list[types.Part] | str, list[dict[str, Any]]]:
        """Process function calls from the Gemini LLM response.

        Args:
            function_calls: Function call objects from LLM response.

        Returns:
            Tuple of (function responses as Parts or error string, file responses).
        """
        try:
            function_text_responses: list[types.Part] = []
            function_file_responses: list[dict[str, Any]] = []

            for tool in function_calls:
                logger.info(f"Processing tool call: {tool.name}")
                logger.info(f"Tool arguments: {tool.args}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(t.name == tool.name for t in tools):
                        try:
                            result = await server.execute_tool(
                                tool.name, tool.args
                            )

                            # Extract content from CallToolResult
                            if isinstance(result, mcp.types.CallToolResult):
                                result = result.content

                            if isinstance(result, list):
                                text_content = None
                                image_content = None

                                for item in result:
                                    if isinstance(item, mcp.types.ImageContent):
                                        image_content = item
                                    if isinstance(item, mcp.types.TextContent):
                                        text_content = item

                                if image_content is not None:
                                    function_file_responses.append({
                                        "type": "image",
                                        "mimeType": image_content.mimeType,
                                        "data": image_content.data,
                                    })

                                if text_content is not None:
                                    logger.debug(
                                        f"Tool call text content: "
                                        f"{text_content.text}"
                                    )
                                    function_response_part = (
                                        types.Part.from_function_response(
                                            name=tool.name,
                                            response={"result": text_content.text}
                                        )
                                    )
                                    function_text_responses.append(
                                        function_response_part
                                    )
                                else:
                                    logger.warning(
                                        "Tool call did not return text content"
                                    )
                            else:
                                function_response_part = (
                                    types.Part.from_function_response(
                                        name=tool.name,
                                        response={"result": result}
                                    )
                                )
                                function_text_responses.append(function_response_part)

                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logger.info(
                                    f"Progress: {progress}/{total} "
                                    f"({percentage:.1f}%)"
                                )

                            return function_text_responses, function_file_responses

                        except Exception as e:
                            error_msg = f"Error executing tool '{tool.name}': {type(e).__name__}: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            return error_msg, function_file_responses

        except Exception as e:
            error_msg = f"Error processing function calls: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, []

        return [], []

    async def send_message(
        self,
        content: str
    ) -> AsyncIterator[str | list[dict[str, Any]]]:
        """Send a message in the chat and stream the response.

        This method implements robust error handling for:
        1. Known Google SDK bug causing JSONDecodeError (Issue #1162)
        2. Server errors (503 UNAVAILABLE, 500 errors) during streaming
        3. Rate limiting (429) during streaming

        Args:
            content: User message content.

        Yields:
            Response chunks (text strings or image data).

        Raises:
            ValueError: If LLM client returns an error.
        """
        max_retries = 3
        retry_delay = 2.0

        for attempt in range(max_retries):
            try:
                llm_response_stream = await self.llm_client.get_response_gemini(
                    types.Part.from_text(text=content)
                )

                if isinstance(llm_response_stream, str):
                    raise ValueError(f"LLM client returned an error: {llm_response_stream}")

                # Process the stream and yield all chunks
                async for chunk in self._process_response_stream(llm_response_stream):
                    yield chunk

                # Success - exit retry loop
                return

            except json.JSONDecodeError as e:
                logger.error(
                    f"JSONDecodeError during streaming (known Google SDK bug): {e}"
                )

                if attempt < max_retries - 1:
                    print(
                        f"\n[Encountered streaming error (Google SDK bug #1162). "
                        f"Retrying... (attempt {attempt + 2}/{max_retries})]"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries exceeded for JSONDecodeError")
                    yield (
                        "\n\n[Error: The response stream was interrupted due to a known "
                        "Google SDK bug. Please try again or rephrase your question. "
                        "Tip: Try disabling thinking mode if enabled.]"
                    )
                    return

            except errors.ServerError as e:
                logger.error(
                    f"ServerError during streaming: Code {e.code}, "
                    f"Status: {e.status}, Message: {e.message}",
                    exc_info=True
                )

                if e.code in [500, 503]:
                    if attempt < max_retries - 1:
                        if e.code == 503:
                            print(
                                f"\n[Server Error {e.code}: {e.message}. "
                                f"Retrying in {retry_delay}s... "
                                f"(attempt {attempt + 2}/{max_retries})]"
                            )
                        else:
                            print(
                                f"\n[Server Error {e.code}: {e.message}. "
                                f"Retrying in {retry_delay}s... "
                                f"(attempt {attempt + 2}/{max_retries})]"
                            )
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"Max retries exceeded for ServerError: {e.message}")
                        yield (
                            f"\n\n[Error: Server error ({e.code}): {e.message}. "
                            f"Please try again or check the logs for details.]"
                        )
                        return
                else:
                    # Other server errors, show specific error and don't retry
                    logger.error(f"Non-retryable ServerError: {e.code} - {e.message}")
                    yield (
                        f"\n\n[Error: Server error ({e.code}): {e.message}]"
                    )
                    return

            except errors.APIError as e:
                logger.error(
                    f"APIError during streaming: Code {e.code}, "
                    f"Message: {e.message}",
                    exc_info=True
                )

                if e.code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        rate_limit_delay = 10.0
                        print(
                            f"\n[API Error {e.code}: {e.message}. "
                            f"Retrying in {rate_limit_delay}s... "
                            f"(attempt {attempt + 2}/{max_retries})]"
                        )
                        await asyncio.sleep(rate_limit_delay)
                    else:
                        logger.error(f"Max retries exceeded for rate limit: {e.message}")
                        yield (
                            f"\n\n[Error: API error ({e.code}): {e.message}. "
                            f"Please wait a moment before trying again.]"
                        )
                        return
                else:
                    # Other API errors, show specific error and don't retry
                    logger.error(f"Non-retryable APIError: {e.code} - {e.message}")
                    yield (
                        f"\n\n[Error: API error ({e.code}): {e.message}]"
                    )
                    return

            except Exception as e:
                # Catch any other unexpected errors
                logger.error(
                    f"Unexpected error during streaming: {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
                if attempt < max_retries - 1:
                    print(
                        f"\n[Unexpected error: {type(e).__name__}. "
                        f"Retrying in {retry_delay}s... "
                        f"(attempt {attempt + 2}/{max_retries})]"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Max retries exceeded for unexpected error: {str(e)}")
                    yield (
                        f"\n\n[Error: {type(e).__name__}: {str(e)}. "
                        f"Check logs with --log-level INFO for details.]"
                    )
                    return

    async def _process_response_stream(
        self,
        llm_response_stream: Any
    ) -> AsyncIterator[str | list[dict[str, Any]]]:
        """Process the response stream with proper error handling.

        Args:
            llm_response_stream: The async stream from Gemini.

        Yields:
            Response chunks (text strings or image data).

        Raises:
            json.JSONDecodeError: If JSON parsing fails (Google SDK bug).
        """

        while True:
            function_call_text_results: list[types.Part] = []
            first_thought = True

            # This try-except will let JSONDecodeError bubble up to the retry logic
            async for llm_response_chunk in llm_response_stream:
                # Handle thinking/thoughts if enabled AND show_thinking is on
                if self.llm_client.enable_thinking and self.llm_client.show_thinking:
                    if llm_response_chunk.candidates:
                        if llm_response_chunk.candidates[0].content:
                            if llm_response_chunk.candidates[0].content.parts:
                                part = llm_response_chunk.candidates[0].content.parts[0]
                                if hasattr(part, 'thought') and part.thought:
                                    if hasattr(part, 'text') and part.text:
                                        if first_thought:
                                            yield f"\n[Thinking:\n{part.text}"
                                            first_thought = False
                                        else:
                                            yield f"{part.text}"
                                    continue

                # Close thinking bracket if thoughts were shown
                if self.llm_client.enable_thinking and self.llm_client.show_thinking and not first_thought:
                    if llm_response_chunk.text or llm_response_chunk.function_calls:
                        yield "]\n\n"
                        first_thought = True  # Reset for next time

                # Show thinking indicator if enabled but not displaying
                if self.llm_client.enable_thinking and not self.llm_client.show_thinking:
                    if llm_response_chunk.candidates:
                        if llm_response_chunk.candidates[0].content:
                            if llm_response_chunk.candidates[0].content.parts:
                                part = llm_response_chunk.candidates[0].content.parts[0]
                                if hasattr(part, 'thought') and part.thought:
                                    if first_thought:
                                        yield "\n[Thinking...]\n\n"
                                        first_thought = False
                                    continue  # Skip the thought content

                # Handle regular text
                if llm_response_chunk.text:
                    yield llm_response_chunk.text

                # Handle function calls
                if llm_response_chunk.function_calls:
                    # Show tool call information
                    for func_call in llm_response_chunk.function_calls:
                        yield f"\n[Tool Call Requested: {func_call.name}]\n"
                        logger.info(f"Tool call requested: {func_call.name}")

                    yield "[Executing tool...]\n"
                    text_results, file_results = await self.process_function_calls(
                        llm_response_chunk.function_calls
                    )

                    # Show success or failure
                    if isinstance(text_results, str):
                        # Error case
                        yield f"[Tool execution failed: {text_results}]\n"
                    else:
                        yield "[Tool executed successfully]\n"

                    if isinstance(text_results, types.Part):
                        function_call_text_results.append(text_results)
                    if isinstance(text_results, list):
                        function_call_text_results.extend(text_results)

                    if file_results and isinstance(file_results, list):
                        logger.info(f"Yielding {len(file_results)} image(s)")
                        for img in file_results:
                            logger.debug(
                                f"Image - type: {img.get('type')}, "
                                f"mimeType: {img.get('mimeType')}, "
                                f"data length: {len(img.get('data', ''))}"
                            )
                        yield file_results

            # Break if no function calls to process
            if not function_call_text_results:
                break

            # Send function results back to LLM
            llm_response_stream = await self.llm_client.get_response_gemini(
                function_call_text_results
            )

            if isinstance(llm_response_stream, str):
                raise ValueError(f"LLM client returned an error: {llm_response_stream}")


class Session:
    """Orchestrates the entire session of interaction between user, chats, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        """Initialize a session.

        Args:
            servers: List of MCP servers to use.
            llm_client: LLM client for communication.
        """
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.chats: list[Chat] = []

    async def initialize_servers(self) -> None:
        """Initialize all servers and list their tools.

        Raises:
            RuntimeError: If server initialization fails.
        """
        for server in self.servers:
            try:
                await server.initialize()
            except Exception as e:
                logger.error(f"Error initializing server {server.name}: {e}")
                await self.cleanup_servers()
                raise RuntimeError(e)

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logger.warning(f"Warning during final cleanup: {e}")

    async def initialize_llm_client(self) -> None:
        """Initialize the LLM client.

        Raises:
            RuntimeError: If LLM client initialization fails.
        """
        try:
            await self.llm_client.start_client_gemini()
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise RuntimeError(e)

    async def new_chat(self) -> Chat:
        """Start a new chat session.

        Returns:
            Initialized Chat object.

        Raises:
            RuntimeError: If chat initialization fails.
        """
        try:
            chat = Chat(self.llm_client, self.servers)
            await chat.start()
            self.chats.append(chat)
            logger.info("New chat session created")
            return chat
        except Exception as e:
            logger.error(f"Error starting new chat: {e}")
            await self.cleanup_servers()
            raise RuntimeError(e)

    async def send_message_to_chat(
        self,
        chat: Chat,
        content: str
    ) -> AsyncIterator[str | list[dict[str, Any]]]:
        """Send a message to the specified chat.

        Args:
            chat: The chat session to send the message to.
            content: Message content.

        Returns:
            Async iterator of response chunks.

        Raises:
            RuntimeError: If sending message fails.
        """
        try:
            response = chat.send_message(content)
            return response
        except Exception as e:
            logger.error(f"Error sending message to chat: {e}")
            raise RuntimeError(e)


def load_questions() -> list[str]:
    """Load questions from questions.txt file.

    Returns:
        List of question strings, one per line.
    """
    questions_file = get_project_root() / "questions.txt"
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(questions)} questions from {questions_file}")
        return questions
    except FileNotFoundError:
        logger.error(f"Questions file not found: {questions_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []


def load_example_indices() -> list[int]:
    """Load example question indices from example_questions.txt file.

    The file should contain comma-separated integers (one-indexed).
    Example: "1,2,3" means use questions 1, 2, and 3 from questions.txt.

    Returns:
        List of zero-indexed integers for questions to use as examples.
    """
    indices_file = get_project_root() / "example_questions.txt"
    try:
        with open(indices_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Parse comma-separated integers (one-indexed)
            one_indexed = [int(x.strip()) for x in content.split(',') if x.strip()]
            # Convert to zero-indexed
            zero_indexed = [i - 1 for i in one_indexed]
            logger.info(f"Example question indices (1-indexed): {one_indexed}")
            logger.info(f"Example question indices (0-indexed): {zero_indexed}")
            return zero_indexed
    except FileNotFoundError:
        logger.warning(f"Example indices file not found: {indices_file}. Using default [0, 1, 2]")
        return [0, 1, 2]  # Default to first 3 questions
    except Exception as e:
        logger.error(f"Error loading example indices: {e}. Using default [0, 1, 2]")
        return [0, 1, 2]


def show_help() -> None:
    """Display help message with available commands."""
    print("\n" + "="*60)
    print("Available Commands:")
    print("="*60)
    print("  /thinking      - Toggle thinking mode on/off")
    print("  /show-thinking - Toggle thinking display on/off")
    print("  /help          - Show this help message")
    print("  /quit or /exit - Exit the application")
    print("="*60 + "\n")


async def main() -> None:
    """Main function to run the session with example questions."""
    # Parse command-line arguments
    args = parse_args()

    # Update log level if specified
    if args.log_level != settings.LOG_LEVEL:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.info(f"Log level set to {args.log_level}")

    session: Optional[Session] = None

    try:
        print("\n" + "="*60)
        print("Singapore Math Bar Model Generator")
        print("="*60)
        print(f"Model: {settings.GEMINI_MODEL}")
        if args.thinking:
            if args.show_thinking:
                print("Thinking mode: enabled (displayed)")
            else:
                print("Thinking mode: enabled (hidden)")
        else:
            print("Thinking mode: disabled")
        print("Type /help for available commands")
        print("="*60 + "\n")

        # Initialize LLM client with thinking mode from args
        # If thinking is enabled but show_thinking is not specified, default to not showing
        # This helps avoid the Google SDK JSON parsing bug
        llm_client = LLMClient(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            max_retries=settings.MAX_RETRIES,
            time_delay=settings.RETRY_DELAY,
            enable_thinking=args.thinking,
            show_thinking=args.show_thinking if args.thinking else False
        )
        await llm_client.start_client_gemini()

        # Load server configuration
        config_path = get_mcp_servers_dir() / "servers_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            server_config = json.load(f)

        servers = [
            Server(name, config)
            for name, config in server_config["mcpServers"].items()
        ]

        # Initialize session with servers and LLM client
        session = Session(servers, llm_client)
        await session.initialize_servers()

        # Start a new chat
        chat = await session.new_chat()

        # Load questions from file
        all_questions = load_questions()
        example_indices = load_example_indices()

        # Get example questions based on indices
        list_qns = []
        if all_questions:
            for idx in example_indices:
                if 0 <= idx < len(all_questions):
                    list_qns.append(all_questions[idx])
                else:
                    logger.warning(f"Example index {idx} out of range (0-{len(all_questions)-1})")

        if not list_qns:
            logger.warning("No example questions available. Using fallback questions.")
            list_qns = [
                "Alice had $20 more than Bob at first. After Alice spent $52, Bob had thrice as much money as Alice. How much money did Alice have at first?",
            ]

        qn_index = 0  # Track which example question we're on
        total_questions = 0  # Track total questions asked (examples + user)
        image_counter = 0  # Track generated images for naming
        use_examples = not args.no_examples

        logger.info(f"Loaded {len(all_questions)} total questions from questions.txt")
        logger.info(f"Using {len(list_qns)} example questions")

        if not use_examples:
            print("Skipping example questions. Enter your questions or type /help.\n")

        while True:
            # Determine if using example or user input
            is_example = use_examples and qn_index < len(list_qns)

            if is_example:
                user_input = list_qns[qn_index]
                qn_index += 1
                total_questions += 1
                input_prefix = f"Question {total_questions}"
                print(f"\n{'='*60}")
                print(f"{input_prefix}: {user_input}")
                print(f"{'='*60}\n")
            else:
                user_input = input("User: ").strip()

                # Handle interactive commands
                if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                    logger.info("Exiting chat session")
                    print("Goodbye!")
                    break

                if user_input.lower() == "/help":
                    show_help()
                    continue

                if user_input.lower() == "/thinking":
                    new_state = llm_client.toggle_thinking_mode()
                    print(f"[Thinking mode: {'enabled' if new_state else 'disabled'}]")
                    # Need to recreate chat session with new thinking config
                    print("[Recreating chat session with new configuration...]")
                    chat = await session.new_chat()
                    print("[Chat session ready]")
                    continue

                if user_input.lower() == "/show-thinking":
                    new_state = llm_client.toggle_show_thinking()
                    if llm_client.enable_thinking:
                        print(f"[Thinking display: {'shown' if new_state else 'hidden'}]")
                        # Need to recreate chat session with new thinking config
                        print("[Recreating chat session with new configuration...]")
                        chat = await session.new_chat()
                        print("[Chat session ready]")
                    else:
                        print("[Warning: Thinking mode is not enabled. Enable it first with /thinking]")
                    continue

                if not user_input:
                    continue

                total_questions += 1

            # Send message to chat and get response
            response = await session.send_message_to_chat(chat, user_input)
            print("Assistant: ")

            async for chunk in response:
                if isinstance(chunk, str):
                    print(chunk, end='', flush=True)
                elif isinstance(chunk, list):
                    # Handle image responses
                    for item in chunk:
                        if item.get("mimeType") == "image/png":
                            images_dir = get_images_dir()
                            image_path = images_dir / f'image_{image_counter:06d}.png'

                            with open(image_path, 'wb') as f:
                                f.write(base64.b64decode(item["data"]))

                            print(f"\n[Image saved to {image_path}]")
                            image_counter += 1  # Increment counter after saving
                            await asyncio.sleep(0.5)

            print("\n")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"\nAn error occurred: {e}")
    finally:
        if session:
            await session.cleanup_servers()
            logger.info("Session cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
