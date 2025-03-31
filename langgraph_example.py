import asyncio
from dataclasses import dataclass
from typing import Annotated, Dict, List, TypedDict
from uuid import uuid4

import uvicorn
from autogen_core import AgentId, MessageContext, RoutedAgent, TopicId, event, rpc
from autogen_ext.runtimes.rest.rest_runtime import RestWorkerAgentRuntime
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, Graph, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from autogen_ext.runtimes.rest.rest_server import app

# Shared message types between AutoGen and LangGraph
@dataclass
class QueryMessage:
    query: str
    context: str = ""

@dataclass
class SearchResult:
    results: List[str]

@dataclass
class SummaryRequest:
    text: str

@dataclass
class SummaryResponse:
    summary: str

# LangGraph state type
class GraphState(TypedDict):
    query: str
    context: str
    results: List[str]
    summary: str

# AutoGen Agents
class SearchAgent(RoutedAgent):
    """Agent that handles search requests."""
    
    def __init__(self) -> None:
        super().__init__("Search agent that finds relevant information")
        # Mock search results for demonstration
        self.search_db = {
            "python": ["Python is a programming language", "Python was created by Guido van Rossum"],
            "autogen": ["AutoGen is a framework for AI agents", "AutoGen supports multi-agent workflows"],
            "langgraph": ["LangGraph enables building agent workflows", "LangGraph is based on LangChain"]
        }

    @rpc
    async def handle_search(self, message: QueryMessage, ctx: MessageContext) -> SearchResult:
        # Simple keyword search
        results = []
        for keyword, entries in self.search_db.items():
            if keyword in message.query.lower():
                results.extend(entries)
        
        return SearchResult(results=results or ["No results found"])

class SummarizerAgent(RoutedAgent):
    """Agent that summarizes information."""

    def __init__(self) -> None:
        super().__init__("Summarizer agent that condenses information")
        self.llm = ChatOpenAI(temperature=0)

    @rpc
    async def handle_summarize(self, message: SummaryRequest, ctx: MessageContext) -> SummaryResponse:
        # Use LLM to summarize
        response = self.llm.invoke([
            HumanMessage(content=f"Please summarize this text concisely: {message.text}")
        ])
        return SummaryResponse(summary=response.content)

    @event
    async def handle_summary_event(self, message: SummaryResponse, ctx: MessageContext) -> None:
        print(f"Received summary event: {message.summary}")

# LangGraph nodes
async def search_node(state: GraphState, runtime: RestWorkerAgentRuntime) -> GraphState:
    """Node that sends search request to AutoGen SearchAgent."""
    # Get search agent
    search_agent_id = await runtime.get("search_agent")
    
    # Send query to search agent
    result = await runtime.send_message(
        message=QueryMessage(query=state["query"], context=state["context"]),
        recipient=search_agent_id
    )
    
    # Update state with results
    state["results"] = result.results
    return state

async def summarize_node(state: GraphState, runtime: RestWorkerAgentRuntime) -> GraphState:
    """Node that sends summarization request to AutoGen SummarizerAgent."""
    # Get summarizer agent
    summarizer_agent_id = await runtime.get("summarizer_agent")
    
    # Combine results into text
    text = "\n".join(state["results"])
    
    # Send summarization request
    result = await runtime.send_message(
        message=SummaryRequest(text=text),
        recipient=summarizer_agent_id
    )
    
    # Update state with summary
    state["summary"] = result.summary
    
    # Also publish summary as event
    await runtime.publish_message(
        message=result,
        topic_id=TopicId("summary", "example")
    )
    
    return state

def should_continue(state: GraphState) -> str:
    """Determine if we should continue processing."""
    return "continue" if not state["summary"] else END

# Example workflow
async def run_workflow():
    """Run a workflow that combines AutoGen and LangGraph."""
    # Create and start runtime
    runtime = RestWorkerAgentRuntime("http://localhost:8000")
    await runtime.start()
    
    try:
        # Register AutoGen agents
        await runtime.register_factory("search_agent", lambda: SearchAgent())
        await runtime.register_factory("summarizer_agent", lambda: SummarizerAgent())
        
        # Create LangGraph workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("search", lambda state: search_node(state, runtime))
        workflow.add_node("summarize", lambda state: summarize_node(state, runtime))
        
        # Add edges
        workflow.add_edge("search", "summarize")
        workflow.add_conditional_edges(
            "summarize",
            should_continue,
            {
                "continue": "search",
                END: END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("search")
        
        # Compile graph
        graph = workflow.compile()
        
        # Run the workflow
        config = {
            "query": "Tell me about Python and AutoGen",
            "context": "",
            "results": [],
            "summary": ""
        }
        
        result = await graph.ainvoke(config)
        print("\nWorkflow result:", result)
        
    finally:
        await runtime.stop()

async def main():
    """Run both server and workflow."""
    # Start server in background
    server_task = asyncio.create_task(run_server())
    
    # Wait for server to start
    await asyncio.sleep(1)
    
    # Run workflow
    await run_workflow()
    
    # Stop server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

async def run_server():
    """Run the FastAPI server."""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main()) 