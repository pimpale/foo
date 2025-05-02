#! /usr/bin/env python3

from mcp import StdioServerParameters
from mcp import ClientSession, StdioServerParameters, stdio_client
import asyncio

async def main():
    # server_params = StdioServerParameters(command="bash", args=["-c", "cd mcp_server && uv run minexample"], env=None)
    server_params = StdioServerParameters(command="docker",  args=['run', '-i', 'minexample', 'uv', '--directory', '/mcp_server', 'run', 'minexample'], env=None)

    # server_params = StdioServerParameters(command="docker",  args=['run', '-i', 'appflowy_taiga', 'uv', '--directory', '/mcp_server', 'run', 'taiga', 'mcp'], env=None)


    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as client_session:

            await client_session.initialize()

            # test tool calls of size 0 to 1_000_000
            for n in range(0, 5_000_000, 10_000):
                result = await client_session.call_tool("make_string", {"n": n})
                print(f"make_string({n}) succeeded")

if __name__ == "__main__":
    asyncio.run(main())