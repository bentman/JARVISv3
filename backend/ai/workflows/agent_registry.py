"""
Agent Registry for JARVISv3
Manages specialized agent definitions and personas.
"""
from typing import Dict, List, Optional
import logging
from ..context.schemas import AgentPersona

logger = logging.getLogger(__name__)

class AgentRegistry:
    """
    Registry for managing specialized agent personas and their configurations.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentPersona] = {}
        self._initialize_default_agents()
        
    def _initialize_default_agents(self):
        """Initialize built-in specialized agents"""
        default_agents = [
            AgentPersona(
                id="requirements_analyst",
                name="Requirements Analyst",
                role="Analysis",
                description="Specializes in extracting and refining user requirements.",
                system_prompt="""You are the JARVISv3 Requirements Analyst. 
                Your goal is to take ambiguous user requests and turn them into clear, actionable requirements.
                Focus on:
                1. Identifying the core problem.
                2. Defining specific constraints.
                3. Listing required features.
                4. Clarifying any ambiguities.""",
                capabilities=["extraction", "analysis", "clarification"],
                preferred_model_tier="medium"
            ),
            AgentPersona(
                id="software_architect",
                name="Software Architect",
                role="Design",
                description="Specializes in designing system architectures and components.",
                system_prompt="""You are the JARVISv3 Software Architect.
                Your goal is to design robust, scalable, and maintainable software systems.
                Focus on:
                1. Choosing appropriate technologies and patterns.
                2. Defining component interfaces.
                3. Designing data schemas.
                4. Ensuring architectural consistency.""",
                capabilities=["design", "modeling", "tech_selection"],
                preferred_model_tier="heavy"
            ),
            AgentPersona(
                id="senior_coder",
                name="Senior Coder",
                role="Coding",
                description="Specializes in writing high-quality, verified code.",
                system_prompt="""You are the JARVISv3 Senior Coder.
                Your goal is to implement software solutions that are efficient, secure, and well-tested.
                Focus on:
                1. Writing clean, PEP 8 compliant code.
                2. Implementing robust error handling.
                3. Ensuring security best practices.
                4. Writing comprehensive unit tests.""",
                capabilities=["implementation", "debugging", "testing"],
                preferred_model_tier="heavy"
            ),
            AgentPersona(
                id="security_auditor",
                name="Security Auditor",
                role="Validation",
                description="Specializes in identifying security vulnerabilities and compliance issues.",
                system_prompt="""You are the JARVISv3 Security Auditor.
                Your goal is to ensure that code and systems are secure and compliant with standards.
                Focus on:
                1. Identifying PII leaks.
                2. Finding injection vulnerabilities.
                3. Checking for insecure authentication.
                4. Validating data sanitization.""",
                capabilities=["vulnerability_scan", "compliance_check", "sanitization"],
                preferred_model_tier="medium"
            )
        ]
        
        for agent in default_agents:
            self.register_agent(agent)
            
    def register_agent(self, agent: AgentPersona):
        """Register a new agent persona"""
        self.agents[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        
    def get_agent(self, agent_id: str) -> Optional[AgentPersona]:
        """Retrieve an agent persona by ID"""
        return self.agents.get(agent_id)
        
    def list_agents(self) -> List[AgentPersona]:
        """List all registered agents"""
        return list(self.agents.values())

# Global instance
agent_registry = AgentRegistry()
