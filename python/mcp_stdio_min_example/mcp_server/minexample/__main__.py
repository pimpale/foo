#! /usr/bin/env python3

import asyncio
import base64
import io
import os

import imageio
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent
from pydantic import Field

mcp = FastMCP("minexample")


# New mcp tool that returns a string of length n
@mcp.tool(
    name="make_string0",
    description="Returns a string of length n composed of 'x' characters"
)
async def make_string0(
    n: int = Field(description="The length of the string to generate"),
) -> str:
    return "x" * n

# New mcp tool that returns a string of length n
@mcp.tool(
    name="make_string",
    description="Returns an 'image' of a string of length n composed of 'x' characters"
)
async def make_string(
    n: int = Field(description="The length of the string to generate"),
) -> CallToolResult:
    randbase64str = base64.b64encode(np.random.randint(0, 256, (n,), dtype=np.uint8)).decode("utf-8")
    return CallToolResult(
        content=[
            ImageContent(
                data=randbase64str
            )
        ]
    )

@mcp.tool(
    name="sleep",
    description="Sleeps for n seconds"
)
async def sleep(
    n: int = Field(description="The number of seconds to sleep"),
) -> str:
    await asyncio.sleep(n)
    return "Done sleeping"


@mcp.tool(
    name="wallpaper_image",
    description="Returns an 'image' of a wallpaper"
)
async def wallpaper_image(
) -> ImageContent:
    # Read wallpaper.jpg, convert to PNG, encode to base64, and return it
    file_path = os.path.join(os.path.dirname(__file__), "..", "wallpaper.jpg")
    img = imageio.imread(file_path)
    buf = io.BytesIO()
    imageio.imwrite(buf, img, format='png')
    buf.seek(0)
    img_bytes = buf.read()
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return ImageContent(type='image', data=encoded, mimeType="image/png")

mcp.run(transport="stdio")