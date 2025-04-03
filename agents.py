import asyncio
from typing import List, Any
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from crewai import Agent as CrewAgent, Task, Crew
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from llama_index.core import Settings
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Configuration
AZURE_CONFIG = {
    "api_key": "your-api-key",
    "api_base": "your-azure-endpoint",
    "api_version": "2024-02-15-preview",
    "deployment_name": "your-deployment"
}

class SimpleAgent(BaseModel):
    name: str
    response: str = ""

class PydanticAIAgent(BaseModel):
    """Pydantic-based AI Agent"""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    llm: Any = Field(None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        self.llm = AzureChatOpenAI(**AZURE_CONFIG)
    
    async def get_response(self, prompt: str) -> str:
        """Get response for a prompt"""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

async def run_all_agents(question: str) -> List[SimpleAgent]:
    """Run all agents with the same question"""
    results = []
    
    # 1. OpenAI Assistant
    async def run_openai():
        client = AzureOpenAI(**AZURE_CONFIG)
        assistant = client.beta.assistants.create(
            name="Simple Assistant",
            instructions="Answer questions concisely",
            model=AZURE_CONFIG["deployment_name"]
        )
        
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=question
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == 'completed':
                break
            await asyncio.sleep(1)
            
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        response = next(msg.content[0].text.value for msg in messages if msg.role == "assistant")
        
        return SimpleAgent(name="OpenAI Assistant", response=response)

    # 2. LangChain/LangGraph
    async def run_langgraph():
        llm = AzureChatOpenAI(**AZURE_CONFIG)
        response = await llm.ainvoke([HumanMessage(content=question)])
        return SimpleAgent(name="LangGraph", response=response.content)

    # 3. CrewAI
    async def run_crewai():
        llm = AzureChatOpenAI(**AZURE_CONFIG)
        agent = CrewAgent(
            role="Assistant",
            goal="Answer questions",
            backstory="I help answer questions",
            llm=llm
        )
        crew = Crew(agents=[agent], tasks=[])
        task = Task(description=question, agent=agent)
        response = await crew.execute_async(tasks=[task])
        return SimpleAgent(name="CrewAI", response=response)

    # 4. LlamaIndex
    async def run_llamaindex():
        llm = LlamaAzureOpenAI(
            model=AZURE_CONFIG["deployment_name"],
            api_key=AZURE_CONFIG["api_key"],
            api_base=AZURE_CONFIG["api_base"],
            api_version=AZURE_CONFIG["api_version"]
        )
        Settings.llm = llm
        response = await llm.acomplete(question)
        return SimpleAgent(name="LlamaIndex", response=response.text)

    # 5. Pydantic AI
    async def run_pydantic():
        agent = PydanticAIAgent(
            name="Pydantic Assistant",
            description="A Pydantic-based AI assistant"
        )
        response = await agent.get_response(question)
        return SimpleAgent(name="Pydantic AI", response=response)

    # Run all agents concurrently
    tasks = [
        run_openai(),
        run_langgraph(),
        run_crewai(),
        run_llamaindex(),
        run_pydantic()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out any errors and return successful results
    return [r for r in results if isinstance(r, SimpleAgent)]

async def main():
    question = "What is the capital of France?"
    results = await run_all_agents(question)
    
    print("\n=== Agent Responses ===")
    for result in results:
        print(f"\n{result.name}:")
        print(result.response)

if __name__ == "__main__":
    asyncio.run(main()) 
