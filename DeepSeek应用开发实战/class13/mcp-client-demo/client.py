import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="uv", # Executable
    args=[
        "run",
        "--with",
        "mcp[cli]",
        "--with-editable",
        "C:\\coding\\project\\ds-test\\DeepSeek应用开发实战\\class12",
        "mcp",
        "run",
        "C:\\coding\\project\\ds-test\\DeepSeek应用开发实战\\class12\\server.py"
    ],# Optional command line arguments
    env=None # Optional environment variables
)


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Tools:", tools)

            # call a tool
            score = await session.call_tool(name="get_score_by_name",arguments={"name": "张三"})
            print("score: ", score)



async def connect_to_sse_server(server_url: str):
    """Connect to an MCP server running with SSE transport"""
    # Store the context managers so they stay alive
    async with sse_client(url=server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # List available tools to verify connection
            print("Initialized SSE client...")
            print("Listing tools...")
            response = await session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])

            # call a tool
            score = await session.call_tool(name="get_score_by_name",arguments={"name": "张三"})

            print("score: ", score)


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    await connect_to_sse_server(server_url=sys.argv[1])


if __name__ == "__main__":
    asyncio.run(main())