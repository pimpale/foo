from contextlib import AsyncExitStack
from mcp import StdioServerParameters
from mcp import ClientSession, StdioServerParameters, stdio_client
import asyncio
from fastapi import FastAPI

app = FastAPI()

class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
    
    async def initialize(self, server_params: StdioServerParameters):

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params),
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

@app.get("/testendpoint")
async def main():
    # server_params = StdioServerParameters(command="bash", args=["-c", "cd mcp_server && uv run minexample"], env=None)
    # server_params = StdioServerParameters(command="docker",  args=['run', '-i', 'minexample', 'uv', '--directory', '/mcp_server', 'run', 'minexample'], env=None)
    server_params = StdioServerParameters(command="bash",  args=['-c', 'docker run -i minexample uv --directory /mcp_server run minexample | tee test.txt'], env=None)


    # server_params = StdioServerParameters(command="docker",  args=['run', '-i', 'appflowy_taiga', 'uv', '--directory', '/mcp_server', 'run', 'taiga', 'mcp'], env=None)

    client = MCPClient()
    print("Initializing client")
    await client.initialize(server_params)

    # test tool call wallpaper_image
    result = await client.session.call_tool("wallpaper_image")
    print(result)

    # # test tool calls of size 0 to 1_000_000
    # for n in range(0, 6_000_000, 10_000):
    #     print(f"make_string({n})")
    #     result = await client.session.call_tool("make_string", {"n": n})
    #     print(f"make_string({n}) succeeded")
