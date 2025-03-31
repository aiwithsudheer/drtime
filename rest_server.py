from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class AgentId(BaseModel):
    type: str
    key: str

class TopicId(BaseModel):
    type: str
    source: str

class Payload(BaseModel):
    data_type: str
    data: Any
    data_content_type: str = "application/json"

class SendMessageRequest(BaseModel):
    request_id: str
    message_id: str
    recipient: AgentId
    sender: Optional[AgentId] = None
    payload: Payload

class PublishMessageRequest(BaseModel):
    message_id: str
    topic: TopicId
    sender: Optional[AgentId] = None
    payload: Payload

class ResponseMessage(BaseModel):
    request_id: str
    payload: Optional[Payload] = None
    error: Optional[str] = None

class Subscription(BaseModel):
    id: str
    topic: TopicId
    recipient: AgentId

class Message(BaseModel):
    type: str = Field(..., regex="^(request|response|event)$")
    request: Optional[SendMessageRequest] = None
    response: Optional[ResponseMessage] = None
    event: Optional[PublishMessageRequest] = None

class RegisterAgentRequest(BaseModel):
    type: str

# Create FastAPI app
app = FastAPI(title="AutoGen REST Runtime Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server state
message_queues: Dict[str, asyncio.Queue[Any]] = defaultdict(asyncio.Queue)
subscriptions: Dict[str, Subscription] = {}
registered_agents: Set[str] = set()

@app.post("/messages/send", status_code=202)
async def send_message(
    request: SendMessageRequest,
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> None:
    """Send a message to an agent."""
    if request.recipient.type not in registered_agents:
        raise HTTPException(status_code=404, detail=f"Agent type {request.recipient.type} not found")

    # Queue message for all clients except sender
    for client_id, queue in message_queues.items():
        if client_id != x_client_id:
            await queue.put({
                "type": "request",
                "request": request.dict()
            })

@app.post("/messages/publish", status_code=202)
async def publish_message(
    request: PublishMessageRequest,
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> None:
    """Publish a message to a topic."""
    # Find matching subscriptions
    matching_recipients = set()
    for sub in subscriptions.values():
        if (sub.topic.type == request.topic.type and 
            sub.topic.source == request.topic.source):
            matching_recipients.add(sub.recipient)

    # Queue message for subscribed clients
    message = {
        "type": "event",
        "event": request.dict()
    }
    
    for client_id, queue in message_queues.items():
        if client_id != x_client_id:
            await queue.put(message)

@app.post("/messages/respond", status_code=202)
async def send_response(
    response: ResponseMessage,
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> None:
    """Send a response to a request."""
    # Queue response for all clients except sender
    for client_id, queue in message_queues.items():
        if client_id != x_client_id:
            await queue.put({
                "type": "response",
                "response": response.dict()
            })

@app.get("/messages/receive")
async def receive_messages(
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> List[Dict[str, Any]]:
    """Long-poll endpoint for receiving messages."""
    try:
        # Get client's message queue
        queue = message_queues[x_client_id]
        
        # Wait for message with timeout
        try:
            message = await asyncio.wait_for(queue.get(), timeout=30.0)
            return [message]
        except asyncio.TimeoutError:
            return []
            
    except Exception as e:
        logger.error(f"Error receiving messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/register", status_code=201)
async def register_agent(
    agent_type: str,
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> None:
    """Register a new agent type."""
    registered_agents.add(agent_type)

@app.post("/subscriptions/add", status_code=201)
async def add_subscription(
    subscription: Subscription,
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> None:
    """Add a new subscription."""
    subscriptions[subscription.id] = subscription

@app.post("/subscriptions/remove", status_code=200)
async def remove_subscription(
    subscription_id: str,
    x_client_id: str = Header(..., alias="X-Client-ID")
) -> None:
    """Remove a subscription."""
    if subscription_id not in subscriptions:
        raise HTTPException(status_code=404, detail="Subscription not found")
    del subscriptions[subscription_id]

# Cleanup handler
@app.on_event("shutdown")
async def cleanup() -> None:
    """Clean up resources when shutting down."""
    message_queues.clear()
    subscriptions.clear()
    registered_agents.clear()

# Create server instance
server = RestRuntimeServer()
app = server.app 