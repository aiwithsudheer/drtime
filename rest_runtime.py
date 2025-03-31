from __future__ import annotations

import asyncio
import json
import logging
import uuid
from asyncio import Future
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Mapping, Set, Type, cast
from urllib.parse import urljoin

import aiohttp
from autogen_core import (
    Agent,
    AgentId,
    AgentMetadata,
    AgentRuntime,
    AgentType,
    CancellationToken,
    MessageContext,
    MessageSerializer,
    Subscription,
    TopicId,
)
from autogen_core._runtime_impl_helpers import SubscriptionManager, get_impl
from autogen_core._serialization import SerializationRegistry

logger = logging.getLogger(__name__)

class RestWorkerAgentRuntime(AgentRuntime):
    """An agent runtime for running distributed agents over HTTP REST APIs.
    
    The runtime communicates with a host server using standard HTTP endpoints for agent messaging
    and management.
    """

    def __init__(
        self,
        host_url: str,
        client_id: str | None = None,
        verify_ssl: bool = True,
    ) -> None:
        """Initialize the REST runtime.

        Args:
            host_url: Base URL of the host server
            client_id: Unique ID for this client. If None, a UUID will be generated
            verify_ssl: Whether to verify SSL certificates
        """
        self._host_url = host_url.rstrip("/") + "/"
        self._client_id = client_id or str(uuid.uuid4())
        self._verify_ssl = verify_ssl
        
        self._agent_factories: Dict[str, Any] = {}
        self._instantiated_agents: Dict[AgentId, Agent] = {}
        self._pending_requests: Dict[str, Future[Any]] = {}
        self._pending_requests_lock = asyncio.Lock()
        self._next_request_id = 0
        self._running = False
        self._client_session: aiohttp.ClientSession | None = None
        self._background_tasks: Set[asyncio.Task[Any]] = set()
        self._subscription_manager = SubscriptionManager()
        self._serialization_registry = SerializationRegistry()

    async def start(self) -> None:
        """Start the runtime and establish connection to host."""
        if self._running:
            raise RuntimeError("Runtime is already running")
            
        self._client_session = aiohttp.ClientSession(
            headers={"X-Client-ID": self._client_id}
        )
        self._running = True
        
        # Start background polling for messages
        self._background_tasks.add(
            asyncio.create_task(self._poll_messages())
        )

    async def stop(self) -> None:
        """Stop the runtime and close connections."""
        if not self._running:
            return
            
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        except Exception as e:
            logger.error("Error stopping runtime", exc_info=e)
            
        # Close HTTP session
        if self._client_session:
            await self._client_session.close()
            self._client_session = None

    async def send_message(
        self,
        message: Any,
        recipient: AgentId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> Any:
        """Send a message to an agent and get response."""
        if not self._running or not self._client_session:
            raise RuntimeError("Runtime not running")

        request_id = await self._get_new_request_id()
        message_id = message_id or str(uuid.uuid4())
        
        # Create future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        # Serialize message
        data_type = self._serialization_registry.type_name(message)
        serialized_message = self._serialization_registry.serialize(
            message, type_name=data_type, data_content_type="application/json"
        )

        # Send request
        payload = {
            "request_id": request_id,
            "message_id": message_id,
            "recipient": {"type": recipient.type, "key": recipient.key},
            "sender": {"type": sender.type, "key": sender.key} if sender else None,
            "payload": {
                "data_type": data_type,
                "data": serialized_message,
                "data_content_type": "application/json"
            }
        }

        async with self._client_session.post(
            urljoin(self._host_url, "messages/send"),
            json=payload,
            verify_ssl=self._verify_ssl
        ) as response:
            if response.status != 202:
                raise RuntimeError(f"Failed to send message: {response.status}")

        return await future

    async def publish_message(
        self,
        message: Any,
        topic_id: TopicId,
        *,
        sender: AgentId | None = None,
        cancellation_token: CancellationToken | None = None,
        message_id: str | None = None,
    ) -> None:
        """Publish a message to a topic."""
        if not self._running or not self._client_session:
            raise RuntimeError("Runtime not running")

        message_id = message_id or str(uuid.uuid4())
        message_type = self._serialization_registry.type_name(message)
        
        # Serialize message
        serialized_message = self._serialization_registry.serialize(
            message, type_name=message_type, data_content_type="application/json"
        )

        # Send publish request
        payload = {
            "message_id": message_id,
            "topic": {"type": topic_id.type, "source": topic_id.source},
            "sender": {"type": sender.type, "key": sender.key} if sender else None,
            "payload": {
                "data_type": message_type,
                "data": serialized_message,
                "data_content_type": "application/json"
            }
        }

        async with self._client_session.post(
            urljoin(self._host_url, "messages/publish"),
            json=payload,
            verify_ssl=self._verify_ssl
        ) as response:
            if response.status != 202:
                raise RuntimeError(f"Failed to publish message: {response.status}")

    async def _poll_messages(self) -> None:
        """Background task to poll for new messages from host."""
        if not self._client_session:
            return
            
        while self._running:
            try:
                async with self._client_session.get(
                    urljoin(self._host_url, "messages/receive"),
                    verify_ssl=self._verify_ssl
                ) as response:
                    if response.status == 200:
                        messages = await response.json()
                        for message in messages:
                            # Process each message in background task
                            task = asyncio.create_task(self._process_message(message))
                            self._background_tasks.add(task)
                            task.add_done_callback(self._background_tasks.discard)
                    
                    elif response.status != 204:  # No content
                        logger.error(f"Error polling messages: {response.status}")
                        
            except Exception as e:
                logger.error("Error polling messages", exc_info=e)
                
            # Wait before next poll
            await asyncio.sleep(1.0)

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """Process an incoming message from host."""
        message_type = message.get("type")
        
        if message_type == "request":
            await self._handle_request(message)
        elif message_type == "response":
            await self._handle_response(message)
        elif message_type == "event":
            await self._handle_event(message)
        else:
            logger.warning(f"Unknown message type: {message_type}")

    async def _handle_request(self, message: Dict[str, Any]) -> None:
        """Handle an incoming request message."""
        request = message["request"]
        recipient = AgentId(request["target"]["type"], request["target"]["key"])
        
        # Get receiving agent
        try:
            agent = await self._get_agent(recipient)
        except ValueError as e:
            await self._send_error_response(request["request_id"], str(e))
            return

        # Create message context
        ctx = MessageContext(
            sender=AgentId(request["source"]["type"], request["source"]["key"]) 
                  if request.get("source") else None,
            topic_id=None,
            is_rpc=True,
            cancellation_token=CancellationToken(),
            message_id=request["request_id"]
        )

        # Handle message
        try:
            # Deserialize message
            message_data = self._serialization_registry.deserialize(
                request["payload"]["data"],
                type_name=request["payload"]["data_type"],
                data_content_type=request["payload"]["data_content_type"]
            )
            
            # Process message
            result = await agent.on_message(message_data, ctx)
            
            # Send response
            await self._send_response(request["request_id"], result)
            
        except Exception as e:
            await self._send_error_response(request["request_id"], str(e))

    async def _handle_response(self, message: Dict[str, Any]) -> None:
        """Handle an incoming response message."""
        response = message["response"]
        request_id = response["request_id"]
        
        if request_id not in self._pending_requests:
            logger.warning(f"Received response for unknown request: {request_id}")
            return
            
        future = self._pending_requests.pop(request_id)
        
        if response.get("error"):
            future.set_exception(Exception(response["error"]))
        else:
            # Deserialize response
            result = self._serialization_registry.deserialize(
                response["payload"]["data"],
                type_name=response["payload"]["data_type"],
                data_content_type=response["payload"]["data_content_type"]
            )
            future.set_result(result)

    async def _handle_event(self, message: Dict[str, Any]) -> None:
        """Handle an incoming event message."""
        event = message["event"]
        topic_id = TopicId(event["topic"]["type"], event["topic"]["source"])
        
        # Get subscribed recipients
        recipients = await self._subscription_manager.get_subscribed_recipients(topic_id)
        
        # Deserialize message
        message_data = self._serialization_registry.deserialize(
            event["payload"]["data"],
            type_name=event["payload"]["data_type"],
            data_content_type=event["payload"]["data_content_type"]
        )

        # Send to all recipients
        sender = AgentId(event["sender"]["type"], event["sender"]["key"]) if event.get("sender") else None
        
        for recipient_id in recipients:
            if recipient_id == sender:
                continue
                
            try:
                agent = await self._get_agent(recipient_id)
                
                ctx = MessageContext(
                    sender=sender,
                    topic_id=topic_id,
                    is_rpc=False,
                    cancellation_token=CancellationToken(),
                    message_id=event["message_id"]
                )
                
                await agent.on_message(message_data, ctx)
                
            except Exception as e:
                logger.error(f"Error delivering event to {recipient_id}", exc_info=e)

    async def _send_response(self, request_id: str, result: Any) -> None:
        """Send a response back to the host."""
        if not self._client_session:
            return
            
        # Serialize result
        result_type = self._serialization_registry.type_name(result)
        serialized_result = self._serialization_registry.serialize(
            result, type_name=result_type, data_content_type="application/json"
        )

        payload = {
            "request_id": request_id,
            "payload": {
                "data_type": result_type,
                "data": serialized_result,
                "data_content_type": "application/json"
            }
        }

        async with self._client_session.post(
            urljoin(self._host_url, "messages/respond"),
            json=payload,
            verify_ssl=self._verify_ssl
        ) as response:
            if response.status != 202:
                logger.error(f"Failed to send response: {response.status}")

    async def _send_error_response(self, request_id: str, error: str) -> None:
        """Send an error response back to the host."""
        if not self._client_session:
            return

        payload = {
            "request_id": request_id,
            "error": error
        }

        async with self._client_session.post(
            urljoin(self._host_url, "messages/respond"),
            json=payload,
            verify_ssl=self._verify_ssl
        ) as response:
            if response.status != 202:
                logger.error(f"Failed to send error response: {response.status}")

    async def _get_new_request_id(self) -> str:
        async with self._pending_requests_lock:
            self._next_request_id += 1
            return str(self._next_request_id)

    async def register_factory(
        self,
        type: str | AgentType,
        agent_factory: Any,
        *,
        expected_class: Type[Agent] | None = None,
    ) -> AgentType:
        """Register an agent factory with the runtime."""
        if isinstance(type, str):
            type = AgentType(type)

        if type.type in self._agent_factories:
            raise ValueError(f"Agent with type {type} already exists")

        self._agent_factories[type.type] = agent_factory

        # Register with host
        if self._client_session:
            async with self._client_session.post(
                urljoin(self._host_url, "agents/register"),
                json={"type": type.type},
                verify_ssl=self._verify_ssl
            ) as response:
                if response.status != 201:
                    raise RuntimeError(f"Failed to register agent type: {response.status}")

        return type

    async def _get_agent(self, agent_id: AgentId) -> Agent:
        """Get or create an agent instance."""
        if agent_id in self._instantiated_agents:
            return self._instantiated_agents[agent_id]

        if agent_id.type not in self._agent_factories:
            raise ValueError(f"Agent with type {agent_id.type} not found")

        factory = self._agent_factories[agent_id.type]
        agent = await factory()
        self._instantiated_agents[agent_id] = agent
        return agent

    async def get(
        self, 
        id_or_type: AgentId | AgentType | str, 
        /, 
        key: str = "default", 
        *, 
        lazy: bool = True
    ) -> AgentId:
        """Get an agent ID."""
        return await get_impl(
            id_or_type=id_or_type,
            key=key,
            lazy=lazy,
            instance_getter=self._get_agent,
        )

    def add_message_serializer(
        self, 
        serializer: MessageSerializer[Any] | Sequence[MessageSerializer[Any]]
    ) -> None:
        """Add message serializers to the runtime."""
        self._serialization_registry.add_serializer(serializer)

    # Not implemented methods
    async def save_state(self) -> Mapping[str, Any]:
        raise NotImplementedError()

    async def load_state(self, state: Mapping[str, Any]) -> None:
        raise NotImplementedError()

    async def agent_metadata(self, agent: AgentId) -> AgentMetadata:
        raise NotImplementedError()

    async def agent_save_state(self, agent: AgentId) -> Mapping[str, Any]:
        raise NotImplementedError()

    async def agent_load_state(self, agent: AgentId, state: Mapping[str, Any]) -> None:
        raise NotImplementedError()

    async def add_subscription(self, subscription: Subscription) -> None:
        """Add a subscription."""
        if not self._client_session:
            raise RuntimeError("Runtime not running")

        # Register with host
        async with self._client_session.post(
            urljoin(self._host_url, "subscriptions/add"),
            json={"subscription": subscription.to_dict()},
            verify_ssl=self._verify_ssl
        ) as response:
            if response.status != 201:
                raise RuntimeError(f"Failed to add subscription: {response.status}")

        # Add locally
        await self._subscription_manager.add_subscription(subscription)

    async def remove_subscription(self, id: str) -> None:
        """Remove a subscription."""
        if not self._client_session:
            raise RuntimeError("Runtime not running")

        # Remove from host
        async with self._client_session.post(
            urljoin(self._host_url, "subscriptions/remove"),
            json={"id": id},
            verify_ssl=self._verify_ssl
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to remove subscription: {response.status}")

        # Remove locally
        await self._subscription_manager.remove_subscription(id) 