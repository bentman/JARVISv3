"""
Context schemas for JARVISv3 - defining the "Golden Context" packets
that flow through the agentic graph system.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, UTC
from enum import Enum


class ContextType(str, Enum):
    """Types of context that can flow through the system"""
    SYSTEM = "system"
    WORKFLOW = "workflow"
    NODE = "node"
    TOOL = "tool"
    USER = "user"
    TASK = "task"


class ModelProfile(BaseModel):
    """Configuration for an LLM model"""
    model_id: str
    filename: str
    profile: str # light, medium, heavy, npu-optimized
    size_mb: int
    description: str


class AgentPersona(BaseModel):
    """Definition of an agent's identity and behavior"""
    id: str
    name: str
    role: str
    description: str
    system_prompt: str
    capabilities: List[str] = []
    preferred_model_tier: str = "medium"


class NodeCapability(BaseModel):
    """Specific capability of a distributed node"""
    hardware_tier: str
    specialized_models: List[str] = []
    max_tokens_capacity: int = 4096
    supports_voice: bool = False


class RemoteNode(BaseModel):
    """Representation of a remote JARVISv3 instance"""
    node_id: str
    name: str
    base_url: str
    status: str = "online" # online, offline, busy
    capabilities: NodeCapability
    current_load: float = 0.0
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HardwareState(BaseModel):
    """Current hardware capabilities and state"""
    gpu_usage: float = Field(ge=0.0, le=100.0, description="GPU utilization percentage")
    memory_available_gb: float
    cpu_usage: float = Field(ge=0.0, le=100.0, description="CPU utilization percentage")
    available_tiers: List[str] = Field(default=["cpu", "gpu", "npu", "cloud"])
    current_load: float = Field(ge=0.0, le=1.0, description="System load factor")


class BudgetState(BaseModel):
    """Current budget and cost tracking"""
    cloud_spend_usd: float = 0.0
    monthly_limit_usd: float = 100.0
    remaining_pct: float = Field(ge=0.0, le=100.0)
    daily_spending: float = 0.0


class UserPreferences(BaseModel):
    """User-specific preferences and settings"""
    preferred_model: Optional[str] = None
    privacy_level: str = "medium"  # low, medium, high
    notification_preferences: Dict[str, bool] = {}
    default_workflow: Optional[str] = None


class SystemContext(BaseModel):
    """Global system-level context"""
    user_id: str
    session_id: str
    hardware_state: HardwareState
    budget_state: BudgetState
    user_preferences: UserPreferences
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskType(str, Enum):
    """Types of tasks that can be processed"""
    CHAT = "chat"
    CODING = "coding"
    RESEARCH = "research"
    AUTOMATION = "automation"
    ANALYSIS = "analysis"
    VOICE = "voice"


class UserIntent(BaseModel):
    """Extracted user intent and task classification"""
    type: TaskType
    confidence: float = Field(ge=0.0, le=1.0)
    description: str
    priority: int = Field(ge=1, le=5)  # 1-5 priority scale


class ContextBudget(BaseModel):
    """Token and resource budgeting for context"""
    max_tokens: int = 100000
    consumed_tokens: int = 0
    remaining_tokens: int = 100000
    max_size_bytes: int = 1000000  # 1MB limit
    current_size_bytes: int = 0
    
    def validate_budget(self) -> bool:
        """Check if context budget is within limits"""
        return (self.consumed_tokens <= self.max_tokens and 
                self.current_size_bytes <= self.max_size_bytes)


class WorkflowContext(BaseModel):
    """Context specific to a workflow execution"""
    workflow_id: str
    workflow_name: str
    initiating_query: str
    user_intent: UserIntent
    accumulated_artifacts: List[str] = []
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    error_history: List[Dict[str, Any]] = []
    human_approvals: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    
    def add_artifact(self, artifact_id: str):
        """Add an artifact to the workflow context"""
        self.accumulated_artifacts.append(artifact_id)
        self.validate_budget()
    
    def add_error(self, error: Dict[str, Any]):
        """Add error to history with limit"""
        self.error_history.append(error)
        if len(self.error_history) > 10:  # Keep only last 10 errors
            self.error_history = self.error_history[-10:]
    
    def validate_budget(self):
        """Validate context budget constraints"""
        if not self.context_budget.validate_budget():
            raise ValueError("Context budget exceeded")


class ToolContext(BaseModel):
    """Context for tool interactions"""
    tools_available: List[str]
    tool_outputs: List[Dict[str, Any]] = []
    tool_execution_history: List[Dict[str, Any]] = []
    permissions: Dict[str, bool] = {}
    
    def add_tool_output(self, tool_name: str, output: Dict[str, Any]):
        """Add tool output to context"""
        self.tool_outputs.append({
            "tool_name": tool_name,
            "output": output,
            "timestamp": datetime.now(UTC).isoformat()
        })


class VoiceContext(BaseModel):
    """Context for voice interactions"""
    audio_input: Optional[bytes] = None  # Raw input audio
    audio_output: Optional[bytes] = None # Generated output audio
    transcription: Optional[str] = None
    voice_id: Optional[str] = "default"
    metadata: Dict[str, Any] = {}


class NodeContext(BaseModel):
    """Context for a specific workflow node"""
    node_id: str
    agent_id: str
    agent_persona: Optional[AgentPersona] = None
    input_context: Dict[str, Any]
    output_context: Optional[Dict[str, Any]] = None
    execution_metadata: Dict[str, Any] = {}
    validation_results: List[Dict[str, Any]] = []
    tokens_consumed: int = 0
    hardware_used: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TaskContext(BaseModel):
    """The 'Golden Context' packet passed between nodes"""
    system_context: SystemContext
    workflow_context: WorkflowContext
    node_context: Optional[NodeContext] = None
    tool_context: Optional[ToolContext] = None
    voice_context: Optional[VoiceContext] = None
    additional_context: Dict[str, Any] = {}
    
    def update_tokens_consumed(self, tokens: int):
        """Update token consumption across all context levels"""
        self.workflow_context.context_budget.consumed_tokens += tokens
        self.workflow_context.context_budget.remaining_tokens -= tokens
        if self.node_context:
            self.node_context.tokens_consumed += tokens
        # Note: validate_budget is called on context_budget, not workflow_context
        if not self.workflow_context.context_budget.validate_budget():
            raise ValueError("Context budget exceeded")
    
    def get_context_size(self) -> int:
        """Calculate approximate context size in bytes"""
        import json
        return len(json.dumps(self.model_dump(), default=str).encode('utf-8'))
    
    def validate_context(self) -> List[str]:
        """Validate the entire context packet"""
        errors = []
        
        # Validate workflow context budget
        try:
            if not self.workflow_context.context_budget.validate_budget():
                errors.append("Context budget exceeded")
        except Exception as e:
            errors.append(f"Budget validation failed: {str(e)}")
        
        # Validate context size
        size = self.get_context_size()
        # Note: max_size_bytes should be compared properly
        if size > self.workflow_context.context_budget.max_size_bytes:
            errors.append(f"Context size {size} exceeds maximum {self.workflow_context.context_budget.max_size_bytes}")
        
        return errors
