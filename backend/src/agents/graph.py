"""LangGraph multi-agent orchestrator: Three-layer architecture.

- Layer 1: Supervisor Agent (receives tasks, plans, delegates, synthesizes)
- Layer 2: Specialist Agents (researcher, data analyst, writer, coder with domain tools)
- Layer 3: Reviewer Agent (validates specialist outputs before returning to the supervisor)
"""

from typing import Any, Dict, List, Literal, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from backend.src.agents.state import (
    AgentState,
    ExecutionPlan,
    ReviewerInput,
    ReviewerOutput,
    SpecialistInput,
    SpecialistOutput,
    SupervisorInput,
    SupervisorOutput,
    SupervisorSynthesisInput,
    SupervisorSynthesisOutput,
)
from backend.src.agents.supervisor.planner import TaskDecompositionEngine
from backend.src.agents.specialist.researcher import ResearchSpecialist
from backend.src.agents.specialist.data_analyst import DataAnalystSpecialist, AnalystSpecialist
from backend.src.agents.specialist.writer import WriterSpecialist
from backend.src.agents.specialist.coder import CoderSpecialist
from backend.src.agents.reviewer.evaluator import ReviewerEvaluator


# ============================================================================
# Router Functions
# ============================================================================

def route_after_planning(state: AgentState) -> str:
    """Evaluate supervisor planning output and route to delegation or escalation."""
    if not state.execution_plan or not state.execution_plan.tasks:
        return "human_escalation"
    return "delegate"


def route_to_assigned_specialist(state: AgentState) -> str:
    """Delegation Router: reads the supervisor's current subtask and delegates

    directly to the corresponding specialist node.
    """
    plan = state.execution_plan
    if not plan or plan.current_task_index >= len(plan.tasks):
        return "synthesize"

    current_task = plan.tasks[plan.current_task_index]
    assigned = current_task.assigned_specialist.lower().strip()

    # Normalize specialist domain names
    if assigned in ("researcher", "research"):
        return "researcher"
    elif assigned in ("analyst", "data_analyst", "data"):
        return "analyst"
    elif assigned in ("coder", "code", "code_execution"):
        return "coder"
    elif assigned in ("writer", "writing"):
        return "writer"

    # Default fallback
    return "researcher"


def route_supervisor_decision(state: AgentState) -> str:
    """Supervisor decision routing after reviewer validation.

    Determines whether to retry, escalate, advance to next specialist, or synthesize.
    """
    plan = state.execution_plan
    if not plan:
        return "human_escalation"

    # If reviewer escalated
    if state.review_status == "escalate":
        return "human_escalation"

    # If reviewer rejected
    if state.review_status == "rejected":
        current_task = plan.tasks[plan.current_task_index]
        retries = state.retry_counts.get(current_task.id, 0)
        if retries < state.max_retries:
            return "retry_specialist"
        return "human_escalation"

    # If approved, check if more tasks remain
    if plan.current_task_index < len(plan.tasks):
        return "next_specialist"

    return "synthesize"


# ============================================================================
# Graph Builder
# ============================================================================

def build_orchestrator_graph(
    supervisor_llm: Optional[BaseChatModel] = None,
    specialist_llm: Optional[BaseChatModel] = None,
    reviewer_llm: Optional[BaseChatModel] = None,
    tools: Optional[Dict[str, List[BaseTool]]] = None,
) -> StateGraph:
    """Build and compile the three-layer LangGraph orchestration workflow.

    Layer 1: Supervisor (TaskDecompositionEngine)
    Layer 2: Specialists (Research, Data Analysis, Writing, Code Execution)
    Layer 3: Reviewer (ReviewerEvaluator)
    """
    tools = tools or {}

    # Initialize Layer 1 (Supervisor) and Layer 3 (Reviewer)
    supervisor = TaskDecompositionEngine(llm=supervisor_llm)
    reviewer = ReviewerEvaluator(llm=reviewer_llm)

    # Initialize Layer 2 (Specialists with domain-specific tools)
    specialists = {
        "researcher": ResearchSpecialist(
            name="researcher",
            llm=specialist_llm,
            tools=tools.get("researcher", []),
        ),
        "analyst": DataAnalystSpecialist(
            name="analyst",
            llm=specialist_llm,
            tools=tools.get("analyst", tools.get("data_analyst", [])),
        ),
        "coder": CoderSpecialist(
            name="coder",
            llm=specialist_llm,
            tools=tools.get("coder", []),
        ),
        "writer": WriterSpecialist(
            name="writer",
            llm=specialist_llm,
            tools=tools.get("writer", []),
        ),
    }

    # ========================================================================
    # Node Definitions with Defined Input/Output Schemas
    # ========================================================================

    # --- Layer 1: Supervisor Planning Node ---
    def supervisor_planning_node(state: AgentState) -> Dict[str, Any]:
        """Supervisor Node (Layer 1): Receives complex task and creates execution plan."""
        # Validate Input Schema
        _ = SupervisorInput(
            task_id=state.task_id or "task_0",
            user_request=state.user_request,
            context={"memories": state.retrieved_memories},
        )

        supervisor_output: SupervisorOutput = supervisor.plan_task(state)

        return {
            "execution_plan": supervisor_output.execution_plan,
            "current_subtask_id": (
                supervisor_output.execution_plan.tasks[0].id
                if supervisor_output.execution_plan.tasks
                else None
            ),
        }

    # --- Layer 2: Specialist Execution Nodes ---
    def make_specialist_node(specialist_key: str):
        """Factory creating LangGraph node for a specialist domain."""
        def specialist_node_fn(state: AgentState) -> Dict[str, Any]:
            plan = state.execution_plan
            if not plan or plan.current_task_index >= len(plan.tasks):
                return {}

            current_task = plan.tasks[plan.current_task_index]
            specialist = specialists[specialist_key]

            # Validate Specialist Input Schema
            feedback_to_pass = (
                state.review_feedback
                if state.review_status == "rejected"
                else None
            )
            spec_input = SpecialistInput(
                subtask_id=current_task.id,
                subtask_description=current_task.description,
                specialist_type=current_task.assigned_specialist,
                inputs=state.specialist_outputs,
                reviewer_feedback=feedback_to_pass,
            )

            # Execute specialist with domain tools
            result: SpecialistOutput = specialist.execute(
                subtask_description=spec_input.subtask_description,
                inputs=spec_input.inputs,
                subtask_id=spec_input.subtask_id,
                reviewer_feedback=spec_input.reviewer_feedback,
            )

            # Store specialist output
            updated_outputs = dict(state.specialist_outputs)
            updated_outputs[current_task.id] = result.output

            return {
                "current_subtask_id": current_task.id,
                "current_specialist_output": result,
                "specialist_outputs": updated_outputs,
            }

        return specialist_node_fn

    # --- Layer 3: Reviewer Node ---
    def reviewer_node(state: AgentState) -> Dict[str, Any]:
        """Reviewer Node (Layer 3): Validates specialist output before returning to supervisor."""
        plan = state.execution_plan
        if not plan or plan.current_task_index >= len(plan.tasks):
            return {"review_status": "approved", "confidence_score": 1.0}

        current_task = plan.tasks[plan.current_task_index]
        specialist_raw_output = state.specialist_outputs.get(current_task.id)

        # Validate Reviewer Input Schema
        _ = ReviewerInput(
            subtask_id=current_task.id,
            subtask_description=current_task.description,
            specialist_type=current_task.assigned_specialist,
            specialist_output=specialist_raw_output,
            expected_format=current_task.expected_output_format,
        )

        return reviewer.evaluate(state)

    # --- Layer 1: Supervisor Decision Node (Reviewer returns to Supervisor) ---
    def supervisor_decision_node(state: AgentState) -> Dict[str, Any]:
        """Supervisor reviews reviewer's validation verdict and coordinates next step.

        If approved: marks task complete and advances task pointer.
        If rejected: updates retry counter for the current subtask.
        """
        plan = state.execution_plan
        if not plan:
            return {"review_status": "escalate"}

        current_task = plan.tasks[plan.current_task_index]
        new_retry_counts = dict(state.retry_counts)

        if state.review_status == "approved":
            current_task.status = "completed"
            current_task.result = state.specialist_outputs.get(current_task.id)
            plan.current_task_index += 1

            next_subtask_id = (
                plan.tasks[plan.current_task_index].id
                if plan.current_task_index < len(plan.tasks)
                else None
            )

            return {
                "execution_plan": plan,
                "current_subtask_id": next_subtask_id,
                "review_feedback": None,
            }

        elif state.review_status == "rejected":
            new_retry_counts[current_task.id] = new_retry_counts.get(current_task.id, 0) + 1
            current_task.status = "rejected"
            return {
                "execution_plan": plan,
                "retry_counts": new_retry_counts,
            }

        else:
            current_task.status = "failed"
            return {"execution_plan": plan}

    # --- Layer 1: Supervisor Delegation Router Node ---
    def supervisor_router_node(state: AgentState) -> Dict[str, Any]:
        """Router node directing control to the assigned specialist."""
        return {}

    # --- Layer 1: Supervisor Final Synthesis Node ---
    def supervisor_synthesize_node(state: AgentState) -> Dict[str, Any]:
        """Supervisor synthesizes all reviewed specialist outputs into final deliverable."""
        synth_output: SupervisorSynthesisOutput = supervisor.synthesize_final_output(state)
        return {"final_output": synth_output.final_output}

    def human_escalation_node(state: AgentState) -> Dict[str, Any]:
        """Fallback node for escalations requiring human intervention."""
        return {
            "review_status": "escalate",
            "final_output": (
                f"Task escalated due to execution issues. Error log: {state.error_log}. "
                f"Last feedback: {state.review_feedback}"
            ),
        }

    # ========================================================================
    # StateGraph Assembly
    # ========================================================================
    workflow = StateGraph(AgentState)

    # Add all layer nodes
    workflow.add_node("supervisor_planner", supervisor_planning_node)
    workflow.add_node("researcher", make_specialist_node("researcher"))
    workflow.add_node("analyst", make_specialist_node("analyst"))
    workflow.add_node("coder", make_specialist_node("coder"))
    workflow.add_node("writer", make_specialist_node("writer"))
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("supervisor_decision", supervisor_decision_node)
    workflow.add_node("synthesize", supervisor_synthesize_node)
    workflow.add_node("human_escalation", human_escalation_node)

    # Entry point is Layer 1: Supervisor
    workflow.set_entry_point("supervisor_planner")

    # Layer 1: Supervisor -> Delegation Router
    workflow.add_conditional_edges(
        "supervisor_planner",
        route_after_planning,
        {
            "delegate": "supervisor_router",  # Dynamic delegation router
            "human_escalation": "human_escalation",
        },
    )

    # Helper router node to dynamically delegate to the assigned specialist
    workflow.add_node("supervisor_router", supervisor_router_node)

    workflow.add_conditional_edges(
        "supervisor_router",
        route_to_assigned_specialist,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "coder": "coder",
            "writer": "writer",
            "synthesize": "synthesize",
        },
    )

    # Layer 2 Specialists -> Layer 3 Reviewer (Always validate before returning)
    for spec_name in ["researcher", "analyst", "coder", "writer"]:
        workflow.add_edge(spec_name, "reviewer")

    # Layer 3 Reviewer -> Layer 1 Supervisor Decision (Returns to supervisor)
    workflow.add_edge("reviewer", "supervisor_decision")

    # Layer 1 Supervisor Decision Routing
    workflow.add_conditional_edges(
        "supervisor_decision",
        route_supervisor_decision,
        {
            "next_specialist": "supervisor_router",
            "retry_specialist": "supervisor_router",
            "synthesize": "synthesize",
            "human_escalation": "human_escalation",
        },
    )

    # Terminal nodes
    workflow.add_edge("synthesize", END)
    workflow.add_edge("human_escalation", END)

    return workflow.compile()
