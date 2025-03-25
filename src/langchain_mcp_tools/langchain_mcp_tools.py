# Standard library imports
from anyio.streams.memory import (
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
)
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Type,
    TypeAlias,
)
import asyncio

# Third-party imports
try:
    from jsonschema_pydantic import jsonschema_to_pydantic  # type: ignore
    from langchain_core.tools import BaseTool, ToolException
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import mcp.types as mcp_types
    from pydantic import BaseModel
    # from pydantic_core import to_json
except ImportError as e:
    print(f'\nError: Required package not found: {e}')
    print('Please ensure all required packages are installed\n')
    sys.exit(1)


def fix_schema(schema: dict) -> dict:
    """Converts JSON Schema 'type': ['string', 'null'] to 'anyOf' format"""
    if isinstance(schema, dict):
        if 'type' in schema and isinstance(schema['type'], list):
            schema['anyOf'] = [{'type': t} for t in schema['type']]
            del schema['type']  # Remove 'type' and standardize to 'anyOf'
        for key, value in schema.items():
            schema[key] = fix_schema(value)  # Apply recursively
    return schema


# Type alias for the bidirectional communication channels with the MCP server
# FIXME: not defined in mcp.types, really?
StdioTransport: TypeAlias = tuple[
    MemoryObjectReceiveStream[mcp_types.JSONRPCMessage | Exception],
    MemoryObjectSendStream[mcp_types.JSONRPCMessage]
]


async def spawn_mcp_server_and_get_transport(
    server_name: str,
    server_config: Dict[str, Any],
    exit_stack: AsyncExitStack,
    timeout: Optional[int] = 60,  # Timeout in seconds
    logger: logging.Logger = logging.getLogger(__name__)
) -> StdioTransport:
    """
    Spawns an MCP server process and establishes communication channels.
    Adds timeout for the connection phase.

    Args:
        timeout: Maximum time in seconds to wait for server initialization.
    """
    try:
        logger.info(f'MCP server "{server_name}": initializing with: {server_config}')

        # NOTE: uv and npx seem to require PATH to be set.
        env = dict(server_config.get('env', {}))
        if 'PATH' not in env:
            env['PATH'] = os.environ.get('PATH', '')

        server_params = StdioServerParameters(
            command=server_config['command'],
            args=server_config.get('args', []),
            env=env
        )

        # Use asyncio.wait_for to apply timeout for the async operation
        stdio_transport = await asyncio.wait_for(
            exit_stack.enter_async_context(stdio_client(server_params)),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.error(f'MCP server "{server_name}" connection timed out after {timeout} seconds')
        raise TimeoutError(f'MCP server "{server_name}" connection timed out after {timeout} seconds')
    except Exception as e:
        logger.error(f'Error spawning MCP server: {str(e)}')
        raise

    return stdio_transport


async def get_mcp_server_tools(
    server_name: str,
    stdio_transport: StdioTransport,
    exit_stack: AsyncExitStack,
    timeout: Optional[int] = 60,  # Timeout in seconds
    logger: logging.Logger = logging.getLogger(__name__)
) -> List[BaseTool]:
    """
    Retrieves and converts MCP server tools to LangChain format.
    Adds timeout for tool retrieval.

    Args:
        timeout: Maximum time in seconds to wait for tools to be retrieved.
    """
    try:
        read, write = stdio_transport

        # Using asyncio.wait_for to apply timeout
        session = await asyncio.wait_for(
            exit_stack.enter_async_context(log_before_aexit(
                ClientSession(read, write),
                f'MCP server "{server_name}": session closed'
            )),
            timeout=timeout
        )

        await session.initialize()
        logger.info(f'MCP server "{server_name}": connected')

        # Get MCP tools with a potential timeout
        tools_response = await asyncio.wait_for(session.list_tools(), timeout=timeout)

        langchain_tools: List[BaseTool] = []
        for tool in tools_response.tools:
            langchain_tools.append(McpToLangChainAdapter())

        logger.info(f'MCP server "{server_name}": {len(langchain_tools)} tool(s) available')

    except asyncio.TimeoutError:
        logger.error(f'MCP server "{server_name}" tool retrieval timed out after {timeout} seconds')
        raise TimeoutError(f'MCP server "{server_name}" tool retrieval timed out after {timeout} seconds')
    except Exception as e:
        logger.error(f'Error getting MCP tools: {str(e)}')
        raise

    return langchain_tools


# Type hint for cleanup function
McpServerCleanupFn = Callable[[], Awaitable[None]]


async def convert_mcp_to_langchain_tools(
    server_configs: Dict[str, Dict[str, Any]],
    logger: logging.Logger = logging.getLogger(__name__)
) -> Tuple[List[BaseTool], McpServerCleanupFn]:
    """Initialize multiple MCP servers and convert their tools to
    LangChain format.

    This async function manages parallel initialization of multiple MCP
    servers, converts their tools to LangChain format, and provides a cleanup
    mechanism. It orchestrates the full lifecycle of multiple servers.

    Args:
        server_configs: Dictionary mapping server names to their
            configurations, where each configuration contains command, args,
            and env settings
        logger: Logger instance to use for logging events and errors.
               Defaults to module logger.

    Returns:
        A tuple containing:
            - List of converted LangChain tools from all servers
            - Async cleanup function to properly shutdown all server
                connections

    Example:
        server_configs = {
            "server1": {"command": "npm", "args": ["start"]},
            "server2": {"command": "./server", "args": ["-p", "8000"]}
        }
        tools, cleanup = await convert_mcp_to_langchain_tools(server_configs)
        # Use tools...
        await cleanup()
    """

    # Initialize AsyncExitStack for managing multiple server lifecycles
    stdio_transports: List[StdioTransport] = []
    async_exit_stack = AsyncExitStack()

    # Spawn all MCP servers concurrently
    for server_name, server_config in server_configs.items():
        # NOTE: the following `await` only blocks until the server subprocess
        # is spawned, i.e. after returning from the `await`, the spawned
        # subprocess starts its initialization independently of (so in
        # parallel with) the Python execution of the following lines.
        stdio_transport = await spawn_mcp_server_and_get_transport(
            server_name,
            server_config,
            async_exit_stack,
            logger
        )
        stdio_transports.append(stdio_transport)

    # Convert tools from each server to LangChain format
    langchain_tools: List[BaseTool] = []
    for (server_name, server_config), stdio_transport in zip(
        server_configs.items(),
        stdio_transports,
        strict=True
    ):
        tools = await get_mcp_server_tools(
            server_name,
            stdio_transport,
            async_exit_stack,
            logger
        )
        langchain_tools.extend(tools)

    # Define a cleanup function to properly shut down all servers
    async def mcp_cleanup() -> None:
        """Closes all server connections and cleans up resources"""
        await async_exit_stack.aclose()

    # Log summary of initialized tools
    logger.info(f'MCP servers initialized: {len(langchain_tools)} tool(s) '
                f'available in total')
    for tool in langchain_tools:
        logger.debug(f'- {tool.name}')

    return langchain_tools, mcp_cleanup
