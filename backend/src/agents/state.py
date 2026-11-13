"""Agent state and LangGraph node input/output schemas."""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


# ============================================================================
# Type Aliases
# ============================================================================
SpecialistType = Literal["researcher", "data_analyst", "analyst", "writer", "coder"]
TaskComplexity = Literal["low", "medium", "high"]
TaskStatus = Literal["pending", "in_progress", "completed", "failed", "rejected"]
ReviewStatus = Literal["approved", "rejected", "escalate"]


# ============================================================================
# Task & Plan Models
# ============================================================================
class SubTask(BaseModel):
    id: str = Field(
        description="Unique identifier for the subtask, e.g. 'subtask_1'",
    )
    description: str = Field(
        description="Detailed, actionable description of what the specialist must accomplish",
    )
    assigned_specialist: SpecialistType = Field(
        description="Domain specialist assigned to execute this subtask: 'researcher', 'data_analyst', 'analyst', 'coder', or 'writer'",
    )
    required_inputs: List[str] = Field(
        default_factory=list,
        description="List of prerequisite subtask IDs (e.g., ['subtask_1']) or input keys needed before execution",
    )
    expected_output_format: str = Field(
        default="text",
        description="Expected format of the deliverable: e.g., 'markdown', 'json', 'python_code', 'text', 'csv'",
    )
    estimated_complexity: TaskComplexity = Field(
        default="medium",
        description="Estimated difficulty and resource cost: 'low', 'medium', or 'high'",
    )
    expected_complexity: Optional[TaskComplexity] = Field(
        default=None,
        description="Alias for estimated_complexity",
    )
    status: TaskStatus = Field(
        default="pending",
        description="Execution status: pending, in_progress, completed, failed, rejected",
    )
    result: Optional[Any] = Field(
        default=None,
        description="Result generated upon subtask completion",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed",
    )

    def model_post_init(self, __context: Any) -> None:
        """Keep estimated_complexity and expected_complexity synchronized."""
        if self.expected_complexity and not self.estimated_complexity:
            object.__setattr__(self, "estimated_complexity", self.expected_complexity)
        elif self.estimated_complexity and not self.expected_complexity:
            object.__setattr__(self, "expected_complexity", self.estimated_complexity)

class ExecutionPlan(BaseModel):
    tasks: List[SubTask] = Field(
        default_factory=list,
        description="Ordered list of sequential, atomic subtasks to be executed by specialists",
    )
    current_task_index: int = Field(
        default=0,
        description="Index of the subtask currently active in execution",
    )
    reasoning: str = Field(
        default="",
        description="Strategic rationale explaining how the task was decomposed and how specialists collaborate",
    )


# Alias for backward compatibility
ExecuationPlan = ExecutionPlan


# ============================================================================
# Defined Input / Output Schemas for LangGraph Nodes
# ============================================================================

# --- Supervisor Layer Schemas ---
class SupervisorInput(BaseModel):
    task_id: str
    user_request: str
    context: Dict[str, Any] = Field(default_factory=dict)


class SupervisorOutput(BaseModel):
    execution_plan: ExecutionPlan
    reasoning: str = ""


class SupervisorSynthesisInput(BaseModel):
    user_request: str
    specialist_outputs: Dict[str, Any] = Field(default_factory=dict)


class SupervisorSynthesisOutput(BaseModel):
    final_output: str


# --- Specialist Layer Schemas ---
class SpecialistInput(BaseModel):
    subtask_id: str
    subtask_description: str
    specialist_type: SpecialistType
    inputs: Dict[str, Any] = Field(default_factory=dict)
    reviewer_feedback: Optional[str] = None


class SpecialistOutput(BaseModel):
    subtask_id: str
    specialist_type: SpecialistType
    status: Literal["success", "error"] = "success"
    output: Any = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


# --- Reviewer Layer Schemas ---
class ReviewerInput(BaseModel):
    subtask_id: str
    subtask_description: str
    specialist_type: SpecialistType
    specialist_output: Any
    expected_format: str = "text"
    criteria: List[str] = Field(default_factory=list)


class ReviewerOutput(BaseModel):
    subtask_id: str
    status: ReviewStatus = "approved"
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    feedback: str = ""
    suggestions: List[str] = Field(default_factory=list)


# ============================================================================
# Global Orchestrator AgentState
# ============================================================================
class AgentState(BaseModel):
    task_id: str = ""
    user_request: str = ""
    session_id: Optional[str] = None

    # Langchain message history
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)

    # Long-term memory context retrieved
    retrieved_memories: List[Dict[str, Any]] = Field(default_factory=list)

    # Execution plan generated by the supervisor (Layer 1)
    execution_plan: Optional[ExecutionPlan] = None

    # Active execution outputs (Layer 2)
    current_subtask_id: Optional[str] = None
    current_specialist_output: Optional[SpecialistOutput] = None
    specialist_outputs: Dict[str, Any] = Field(default_factory=dict)

    # Reviewer validation & confidence tracking (Layer 3)
    review_status: ReviewStatus = "approved"
    review_feedback: Optional[str] = None
    confidence_score: float = 1.0
    reviewer_output: Optional[ReviewerOutput] = None

    # Error management & retry loops
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    max_retries: int = 2
    error_log: List[str] = Field(default_factory=list)

    # Final synthesized deliverable
    final_output: Optional[str] = None
