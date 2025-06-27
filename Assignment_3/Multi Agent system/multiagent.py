import os
import time
import uuid
from abc import ABC, abstractmethod
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
from google import genai
from google.genai import types

# Message Types and Structure
class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_SHARE = "information_share"
    COORDINATION = "coordination"

@dataclass
class Message:
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    message_id: str

# Base Agent Class with Gemini Integration
class GeminiAgent(ABC):
    def __init__(self, name: str, model: str = "gemini-2.0-flash-001"):
        self.name = name
        self.model = model
        self.inbox = Queue()
        self.memory = []
        self.is_running = False
        
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)
        
    def generate_response(self, prompt: str, system_instruction: str = None) -> str:
        """Generate response using Gemini API"""
        try:
            if system_instruction:
                full_prompt = f"System: {system_instruction}\n\nUser: {prompt}"
            else:
                full_prompt = prompt
                
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1000
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def send_message(self, receiver: 'GeminiAgent', message_type: MessageType, content: dict):
        """Send a message to another agent"""
        message = Message(
            sender=self.name,
            receiver=receiver.name,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )
        receiver.receive_message(message)
        self.memory.append(f"Sent {message_type.value} to {receiver.name}")
        
    def receive_message(self, message: Message):
        """Receive a message from another agent"""
        self.inbox.put(message)
        
    @abstractmethod
    def process_message(self, message: Message):
        """Process incoming messages - to be implemented by subclasses"""
        pass
        
    def start(self):
        """Start the agent's message processing loop"""
        self.is_running = True
        self.processing_thread = Thread(target=self._message_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def _message_loop(self):
        """Main message processing loop"""
        while self.is_running:
            if not self.inbox.empty():
                message = self.inbox.get()
                response = self.process_message(message)
                if response:
                    # Send response back if needed
                    pass
            time.sleep(0.1)

# Planning Agent with Gemini
class GeminiPlanningAgent(GeminiAgent):
    def __init__(self):
        super().__init__("PlanningAgent", "gemini-2.0-flash-001")
        self.system_instruction = """You are a planning agent in a multi-agent system. 
        Your role is to break down complex queries into structured, actionable steps.
        Always respond with a clear plan in JSON format with numbered steps."""
        
    def process_message(self, message: Message):
        if message.message_type == MessageType.TASK_REQUEST:
            return self._create_plan(message.content.get("query", ""))
        return None
        
    def _create_plan(self, query: str) -> Message:
        """Create a structured plan using Gemini"""
        prompt = f"""
        Create a detailed plan to answer this query: "{query}"
        
        Provide your response as a structured plan with:
        1. Query analysis
        2. Required research steps
        3. Information synthesis approach
        4. Expected deliverables
        
        Format as clear, actionable steps.
        """
        
        plan_text = self.generate_response(prompt, self.system_instruction)
        
        plan_content = {
            "plan_id": str(uuid.uuid4()),
            "original_query": query,
            "plan_details": plan_text,
            "status": "created",
            "agent": self.name
        }
        
        self.memory.append(f"Created plan for query: {query}")
        
        return Message(
            sender=self.name,
            receiver="CoordinatorAgent",
            message_type=MessageType.TASK_RESPONSE,
            content=plan_content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )

# Research Agent with Real-time Search
class GeminiResearchAgent(GeminiAgent):
    def __init__(self):
        super().__init__("ResearchAgent", "gemini-2.0-flash-001")
        self.system_instruction = """You are a research agent specialized in gathering 
        and analyzing information. Provide comprehensive, well-sourced research findings."""
        
    def process_message(self, message: Message):
        if message.message_type == MessageType.TASK_REQUEST:
            return self._conduct_research(message.content)
        return None
        
    def _conduct_research(self, task_content: dict) -> Message:
        """Conduct research using Gemini with real-time search"""
        query = task_content.get("query", "")
        
        # Use Gemini with Google Search tool for real-time information
        search_tool = {'google_search': {}}
        
        try:
            chat = self.client.chats.create(
                model=self.model, 
                config={'tools': [search_tool]}
            )
            
            research_prompt = f"""
            Research the following topic comprehensively: "{query}"
            
            Please provide:
            1. Current information and recent developments
            2. Key facts and statistics
            3. Different perspectives or viewpoints
            4. Relevant examples or case studies
            5. Summary of findings with confidence assessment
            """
            
            response = chat.send_message(research_prompt)
            research_findings = response.candidates[0].content.parts[0].text
            
        except Exception as e:
            # Fallback to regular generation if search fails
            research_findings = self.generate_response(
                f"Provide comprehensive research on: {query}", 
                self.system_instruction
            )
        
        research_content = {
            "query": query,
            "findings": research_findings,
            "research_method": "gemini_with_search",
            "confidence_score": 0.9,
            "agent": self.name
        }
        
        self.memory.append(f"Completed research for: {query}")
        
        return Message(
            sender=self.name,
            receiver="CoordinatorAgent",
            message_type=MessageType.TASK_RESPONSE,
            content=research_content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )

# Summarization Agent with Structured Output
class GeminiSummarizationAgent(GeminiAgent):
    def __init__(self):
        super().__init__("SummarizationAgent", "gemini-2.0-flash-001")
        self.system_instruction = """You are a summarization agent that creates 
        clear, comprehensive summaries. Always structure your output with clear sections."""
        
    def process_message(self, message: Message):
        if message.message_type == MessageType.TASK_REQUEST:
            return self._create_summary(message.content)
        return None
        
    def _create_summary(self, content: dict) -> Message:
        """Create summary using Gemini with structured output"""
        research_data = content.get("research_findings", "")
        original_query = content.get("original_query", "")
        plan_details = content.get("plan_details", "")
        
        summary_prompt = f"""
        Based on the research findings below, create a comprehensive summary for the original query.
        
        Original Query: {original_query}
        Plan: {plan_details}
        Research Findings: {research_data}
        
        Provide a structured summary with:
        1. Executive Summary (2-3 sentences)
        2. Key Findings (bullet points)
        3. Detailed Analysis
        4. Conclusions and Recommendations
        5. Confidence Assessment
        
        Make it clear, actionable, and well-organized.
        """
        
        # Use structured output generation
        summary_text = self.generate_response(summary_prompt, self.system_instruction)
        
        summary_content = {
            "original_query": original_query,
            "summary": summary_text,
            "summary_type": "comprehensive",
            "agent": self.name,
            "timestamp": time.time()
        }
        
        self.memory.append(f"Generated summary for: {original_query}")
        
        return Message(
            sender=self.name,
            receiver="CoordinatorAgent",
            message_type=MessageType.TASK_RESPONSE,
            content=summary_content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )

# Enhanced Coordinator with Gemini
class GeminiCoordinatorAgent(GeminiAgent):
    def __init__(self, agents: dict):
        super().__init__("CoordinatorAgent", "gemini-2.0-flash-001")
        self.agents = agents
        self.active_tasks = {}
        self.system_instruction = """You are a coordinator agent managing a multi-agent system.
        Your role is to orchestrate tasks between specialized agents and synthesize final responses."""
        
    def process_user_query(self, query: str) -> str:
        """Process user query through the multi-agent system"""
        task_id = str(uuid.uuid4())
        print(f"🎯 Processing query: {query}")
        print(f"📋 Task ID: {task_id}")
        
        self.active_tasks[task_id] = {
            "query": query,
            "status": "planning",
            "results": {},
            "start_time": time.time()
        }
        
        # Step 1: Planning
        print("📝 Step 1: Creating plan...")
        planning_response = self._get_agent_response("planning", {
            "query": query, 
            "task_id": task_id
        })
        
        # Step 2: Research
        print("🔍 Step 2: Conducting research...")
        research_response = self._get_agent_response("research", {
            "query": query,
            "task_id": task_id,
            "plan": planning_response.content if planning_response else None
        })
        
        # Step 3: Summarization
        print("📊 Step 3: Creating summary...")
        summary_response = self._get_agent_response("summarization", {
            "original_query": query,
            "plan_details": planning_response.content.get("plan_details", "") if planning_response else "",
            "research_findings": research_response.content.get("findings", "") if research_response else "",
            "task_id": task_id
        })
        
        # Generate final response
        final_response = self._generate_final_response(query, summary_response)
        
        print(f"✅ Task {task_id} completed!")
        return final_response
        
    def _get_agent_response(self, agent_type: str, content: dict) -> Message:
        """Get response from a specific agent"""
        if agent_type in self.agents:
            agent = self.agents[agent_type]
            message = Message(
                sender=self.name,
                receiver=agent.name,
                message_type=MessageType.TASK_REQUEST,
                content=content,
                timestamp=time.time(),
                message_id=str(uuid.uuid4())
            )
            return agent.process_message(message)
        return None
        
    def _generate_final_response(self, query: str, summary_response: Message) -> str:
        """Generate final coordinated response"""
        if summary_response and summary_response.content:
            summary_text = summary_response.content.get("summary", "")
            
            final_prompt = f"""
            Based on the comprehensive analysis below, provide a final, polished response to the user's query.
            
            Original Query: {query}
            Analysis Summary: {summary_text}
            
            Provide a clear, direct answer that addresses the user's question while incorporating 
            the key insights from the analysis. Make it conversational and helpful.
            """
            
            final_response = self.generate_response(final_prompt, self.system_instruction)
            return final_response
        else:
            return f"I apologize, but I encountered an issue processing your query: {query}"
    
    def process_message(self, message: Message):
        """Handle responses from other agents"""
        if message.message_type == MessageType.TASK_RESPONSE:
            self._handle_agent_response(message)
        return None
        
    def _handle_agent_response(self, message: Message):
        """Process responses from specialized agents"""
        sender = message.sender
        content = message.content
        
        self.memory.append(f"Received response from {sender}")
        
        # Update task status based on sender
        if sender == "PlanningAgent":
            print(f"✓ Plan received: Plan created successfully")
        elif sender == "ResearchAgent":
            print(f"✓ Research completed: {content.get('confidence_score', 0)} confidence")
        elif sender == "SummarizationAgent":
            print(f"✓ Summary generated: {content.get('summary_type', 'unknown')} type")

# System Integration and Usage
def create_gemini_multi_agent_system():
    """Initialize the Gemini-powered multi-agent system"""
    
    # Verify API key
    if not os.environ.get("GEMINI_API_KEY"):
        raise ValueError("Please set GEMINI_API_KEY environment variable")
    
    # Create specialized agents
    planning_agent = GeminiPlanningAgent()
    research_agent = GeminiResearchAgent()
    summarization_agent = GeminiSummarizationAgent()
    
    # Create agent registry
    agents = {
        "planning": planning_agent,
        "research": research_agent,
        "summarization": summarization_agent
    }
    
    # Create coordinator
    coordinator = GeminiCoordinatorAgent(agents)
    
    # Start all agents
    for agent in agents.values():
        agent.start()
    coordinator.start()
    
    return coordinator, agents

def main():
    """Demonstrate the Gemini-powered multi-agent system"""
    print("🚀 Initializing Gemini Multi-Agent System...")
    
    try:
        coordinator, agents = create_gemini_multi_agent_system()
        
        # Example queries
        queries = [
            "What are the latest developments in quantum computing?",
            "How is artificial intelligence transforming healthcare?",
            "What are the environmental impacts of renewable energy adoption?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"🔄 Query {i}/3")
            print(f"{'='*80}")
            
            result = coordinator.process_user_query(query)
            
            print(f"\n📋 Final Response:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
            if i < len(queries):
                print("\n⏳ Waiting before next query...")
                time.sleep(3)
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure your GEMINI_API_KEY is set correctly.")

def interactive_mode():
    """Run the system in interactive mode"""
    print("🚀 Initializing Gemini Multi-Agent System...")
    
    try:
        coordinator, agents = create_gemini_multi_agent_system()
        
        print("System ready! Type 'quit' to exit.")
        
        while True:
            query = input("\n📝 Enter your query: ")
            if query.lower() == 'quit':
                break
                
            result = coordinator.process_user_query(query)
            print(f"\n✅ Response:\n{result}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please ensure your GEMINI_API_KEY is set correctly.")

if __name__ == "__main__":
    # Choose mode: main() for demo or interactive_mode() for interactive use
    main()
    # Uncomment the line below to run in interactive mode instead
    # interactive_mode()
