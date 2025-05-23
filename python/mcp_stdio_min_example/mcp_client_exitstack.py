#! /usr/bin/env python3

from contextlib import AsyncExitStack
from mcp import StdioServerParameters
from mcp import ClientSession, StdioServerParameters, stdio_client
import asyncio


class TestMCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self):
        # server_params = StdioServerParameters(command="bash", args=["-c", "cd mcp_server && uv run minexample"], env=None)
        server_params = StdioServerParameters(
            command="docker",
            args=["run", "-i", "minexample", "uv", "--directory", "/mcp_server", "run", "minexample"],
            env=None,
        )

      
        server_params = StdioServerParameters(command="bash",  args=['-c', 'docker run --network none -i appflowy_taiga uv --directory /mcp_server run taiga mcp | tee yeet.txt'], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params),
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()


async def main():
    client = TestMCPClient()
    print("Initializing client")
    await client.connect_to_server()


    result0 = await client.session.call_tool("setup_problem", {"problem_id": "write-hello-world"})
    print(result0)
    result = await client.session.call_tool("computer_use", {"action": "screenshot", 'kwargs': ''})
    print(result)

    # test tool call wallpaper_image
    result = await client.session.call_tool("wallpaper_image")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())