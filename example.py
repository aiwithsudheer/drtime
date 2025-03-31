import asyncio
import uvicorn
from dataclasses import dataclass
from typing import Optional

from autogen_core import MessageContext, RoutedAgent, event, rpc
from autogen_ext.runtimes.rest.rest_runtime import RestWorkerAgentRuntime

# Example message types
@dataclass
class Greeting:
    message: str

@dataclass
class Response:
    content: str

# Example agent implementation
class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("Example agent that handles greetings")

    @rpc
    async def handle_greeting(self, message: Greeting, ctx: MessageContext) -> Response:
        return Response(f"Received greeting: {message.message}")

    @event
    async def handle_broadcast(self, message: Greeting, ctx: MessageContext) -> None:
        print(f"Received broadcast: {message.message}")

async def run_server():
    """Run the FastAPI server."""
    config = uvicorn.Config("example:app", host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def run_client():
    """Run a sample client."""
    # Create and start runtime
    runtime = RestWorkerAgentRuntime("http://localhost:8000")
    await runtime.start()
    
    try:
        # Register agent factory
        agent_type = await runtime.register_factory("greeting_agent", lambda: MyAgent())
        
        # Get agent ID
        agent_id = await runtime.get(agent_type)
        
        # Send RPC message
        response = await runtime.send_message(
            message=Greeting("Hello!"),
            recipient=agent_id
        )
        print(f"Got response: {response}")
        
        # Publish event message
        await runtime.publish_message(
            message=Greeting("Broadcast message"),
            topic_id=TopicId("greeting", "example")
        )
        
    finally:
        await runtime.stop()

async def main():
    """Run both server and client."""
    # Start server in background
    server_task = asyncio.create_task(run_server())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    # Run client
    await run_client()
    
    # Stop server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main()) 