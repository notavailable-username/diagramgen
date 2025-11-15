"""MCP Server for Singapore Math Bar Model Generation.

This module implements a Model Context Protocol (MCP) server that provides
tools and prompts for generating bar model visualizations for Singapore
primary school mathematics word problems.
"""

import anyio
import base64
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Any, List, Optional

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from bar_model import start_canvas, BarModel, save_image


# Initialize MCP server
app = Server("pri-math-model")
_tool_functions_map: dict[str, Any] = {}


def create_bar_model(
    bars_data: list[dict[str, Any]],
    v_braces_data: Optional[list[dict[str, Any]]] = None
) -> list[types.TextContent | types.ImageContent]:
    """Create a bar model visualization with the given bars data and optional vertical braces data.

    This tool generates a bar model diagram commonly used in Singapore Math
    to visualize word problems. It creates horizontal bars with segments,
    horizontal braces for labeling groups of segments, and vertical braces
    for showing relationships between bars.

    Args:
        bars_data: List of bar objects. Each bar contains:
            - segments: List of dicts with 'length' (relative) and 'label' fields
            - h_braces: List of horizontal brace dicts with 'start_segment_index',
                       'end_segment_index', 'label', and 'location' ('top'/'bottom')
            - label: Label for the entire bar (displayed on the left)
        v_braces_data: Optional list of vertical brace objects with:
            - start_bar_index: Index of starting bar (0-indexed)
            - end_bar_index: Index of ending bar (0-indexed)
            - label: Label for the vertical brace

    Returns:
        List containing two elements:
            - types.TextContent: Success message
            - types.ImageContent: Base64-encoded PNG image of the bar model

    Raises:
        Exception: If the bar model cannot be created.
    """
    # Convert dicts to tuples for segments, h_braces, and v_braces_data
    def convert_bar(bar: dict[str, Any]) -> dict[str, Any]:
        """Convert bar dictionary to expected format with tuples."""
        segments = [
            (seg["length"], seg["label"])
            for seg in bar.get("segments", [])
        ]
        h_braces = [
            (
                brace["start_segment_index"],
                brace["end_segment_index"],
                brace["label"],
                brace["location"]
            )
            for brace in bar.get("h_braces", [])
        ]
        return {
            "segments": segments,
            "h_braces": h_braces,
            "label": bar.get("label", "")
        }

    bars_data = [convert_bar(bar) for bar in bars_data]

    if v_braces_data is not None:
        v_braces_data = [
            (
                brace["start_bar_index"],
                brace["end_bar_index"],
                brace["label"]
            )
            for brace in v_braces_data
        ]

    try:
        # Initialize canvas with fixed size
        canvas = start_canvas(2000, 1000)

        # Create the bar model
        model = BarModel(canvas, bars_data, v_braces_data)

        # Get the canvas with drawn elements
        result_canvas = model.get_canvas()

        # Convert to PNG and encode as base64
        _, buffer = cv2.imencode('.png', result_canvas)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return [
            types.TextContent(
                type="text",
                text="Image generated successfully."
            ),
            types.ImageContent(
                type="image",
                data=img_base64,
                mimeType="image/png"
            )
        ]
    except Exception as e:
        raise Exception(f"Failed to create bar model: {str(e)}")


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """List available prompts from the MCP server.

    Returns:
        List of available prompt definitions.
    """
    return [
        types.Prompt(
            name="system-prompt",
            description="System instructions for the assistant."
        ),
        types.Prompt(
            name="bar-model-drawing",
            description="Instructions for solving math word problems with bar models."
        )
    ]


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools from the MCP server.

    This function defines all tools that can be called by clients.
    Each tool is registered with its name, description, and input schema.

    Returns:
        List of available tool definitions.
    """
    global _tool_functions_map
    _tool_functions_map = {}  # Reset for idempotency

    # Define base tools with their actual function references
    base_tool_definitions = [
        {
            "name": "create_bar_model",
            "description": (
                "This tool generates a bar model visualization based on provided data. "
                "It can create multiple bars, each with its own segments, labels, and horizontal braces. "
                "Horizontal braces and segments are good for representing difference or proportion relationships. "
                "Horizontal braces can span across one or more segments of a bar, and their span can overlap with the span of other braces. "
                "It also supports vertical braces that can span across multiple bars. "
                "Vertical braces should only be used to represent sum relationships across multiple bars. "
                "The function returns a list with two elements: "
                "[1] types.TextContent (a text message indicating success), and "
                "[2] types.ImageContent (the generated bar model as a base64-encoded PNG image, mimeType=\"image/png\"). "
                "All indices for bars and segments are 0-indexed. The drawing logic handles layout, font scaling, and brace placement to create a clear and readable diagram."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["bars_data"],
                "properties": {
                    "bars_data": {
                        "type": "array",
                        "description": "List of bar objects. Each object represents a horizontal bar in the model.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segments": {
                                    "type": "array",
                                    "description": "A list of segments that make up the bar. Each segment is an object with 'length' (relative length across all bars) and 'label' fields.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "length": {
                                                "type": "number",
                                                "description": "Relative length of the segment with respect to other segments across all bars."
                                            },
                                            "label": {
                                                "type": "string",
                                                "description": "Label for the segment."
                                            }
                                        },
                                        "required": ["length", "label"]
                                    }
                                },
                                "h_braces": {
                                    "type": "array",
                                    "description": "Horizontal (curly) braces that can span across one or more segments of this bar. Each brace is an object with 'start_segment_index', 'end_segment_index', 'label', and 'location' fields.",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "start_segment_index": {
                                                "type": "integer",
                                                "description": "Start segment index."
                                            },
                                            "end_segment_index": {
                                                "type": "integer",
                                                "description": "End segment index."
                                            },
                                            "label": {
                                                "type": "string",
                                                "description": "Label for the brace."
                                            },
                                            "location": {
                                                "type": "string",
                                                "description": "Location ('top' or 'bottom')."
                                            }
                                        },
                                        "required": ["start_segment_index", "end_segment_index", "label", "location"]
                                    }
                                },
                                "label": {
                                    "type": "string",
                                    "description": "A label for the entire bar. This is drawn to the left of the respective bar."
                                }
                            }
                        }
                    },
                    "v_braces_data": {
                        "type": "array",
                        "description": "A list of vertical braces that can span across multiple bars. Each brace is an object with 'start_bar_index', 'end_bar_index', and 'label' fields. Vertical braces should only be used to represent sum relationships across multiple bars.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_bar_index": {
                                    "type": "integer",
                                    "description": "Start bar index."
                                },
                                "end_bar_index": {
                                    "type": "integer",
                                    "description": "End bar index."
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Label for the vertical brace."
                                }
                            },
                            "required": ["start_bar_index", "end_bar_index", "label"]
                        }
                    }
                }
            },
            "function": create_bar_model
        },
    ]

    TOOLS: List[types.Tool] = []
    for i, tool_def in enumerate(base_tool_definitions):
        tool_id = i + 1
        tool_alias = f"f{tool_id}"
        TOOLS.append(
            types.Tool(
                name=tool_def["name"],
                description=tool_def["description"],
                inputSchema=tool_def["inputSchema"]
            )
        )
        _tool_functions_map[tool_alias] = tool_def["function"]

    return TOOLS


@app.get_prompt()
async def get_prompt(
    name: str,
    arguments: Optional[dict[str, str]] = None
) -> types.GetPromptResult:
    """Get a prompt by name with optional arguments.

    Args:
        name: Name of the prompt to retrieve.
        arguments: Optional arguments for the prompt (currently unused).

    Returns:
        GetPromptResult containing the prompt content.

    Raises:
        ValueError: If the prompt file is not found.
    """
    prompt_file_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        f"{name}.txt"
    )

    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        raise ValueError(f"Prompt file not found for '{name}'")

    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=prompt_text
                )
            )
        ],
        description=f"Instructions for {name}."
    )


@app.call_tool()
async def call_tool(
    name: str,
    arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Call a tool by name with arguments.

    Args:
        name: Name of the tool to call.
        arguments: Dictionary of tool arguments.

    Returns:
        List of content items (text, images, or embedded resources).
    """
    if name == "add":
        a = arguments.get("a")
        b = arguments.get("b")

        # Validate arguments
        if not isinstance(a, int) or not isinstance(b, int):
            return [
                types.TextContent(
                    type="text",
                    text="Invalid arguments. 'a' and 'b' must be integers."
                )
            ]

        result = add(a=a, b=b)
        return [
            types.TextContent(
                type="text",
                text=f"{result}"
            )
        ]

    if name == "create_bar_model":
        bars_data = arguments.get("bars_data", [])
        v_braces_data = arguments.get("v_braces_data", None)

        try:
            result = create_bar_model(bars_data, v_braces_data)
            return result
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error creating bar model: {str(e)}"
                )
            ]

    # Tool not found
    return [
        types.TextContent(
            type="text",
            text=f"Error: Tool '{name}' not found or not implemented."
        )
    ]


async def arun() -> None:
    """Run the MCP server with stdio transport."""
    async with stdio_server() as streams:
        await app.run(
            streams[0],
            streams[1],
            app.create_initialization_options()
        )


if __name__ == "__main__":
    anyio.run(arun)
