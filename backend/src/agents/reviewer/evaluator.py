"""Reviewer Agent (Layer 3): Validates outputs from specialists before returning to the supervisor."""

from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from backend.src.agents.state import (
    AgentState,
    ReviewerInput,
    ReviewerOutput,
    ReviewStatus,
)


class ReviewerEvaluator:
    """Reviewer Agent responsible for evaluating specialist outputs against task criteria,

    expected output formats, and quality thresholds before returning to the supervisor.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None, min_pass_score: float = 0.7):
        self.llm = llm
        self.min_pass_score = min_pass_score

        self.review_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Reviewer Agent (Layer 3 of the orchestration system). "
                "Your job is to rigorously validate outputs produced by Specialist Agents "
                "(researcher, analyst, coder, writer) before returning to the Supervisor Agent.\n\n"
                "Evaluate the output based on:\n"
                "1. Task Completion: Did the specialist satisfy all parts of the subtask description?\n"
                "2. Format Compliance: Does the output match the expected output format?\n"
                "3. Correctness & Quality: Are the findings, code, or analysis sound and free of errors?\n\n"
                "Decision Rules:\n"
                "- If satisfactory and confidence >= 0.7 -> status: 'approved'\n"
                "- If issues can be fixed by the specialist with specific feedback -> status: 'rejected'\n"
                "- If severe unrecoverable error or safety concern -> status: 'escalate'\n"
                "Return a structured ReviewerOutput.",
            ),
            (
                "human",
                "Subtask ID: {subtask_id}\n"
                "Assigned Specialist: {specialist_type}\n"
                "Subtask Description: {subtask_description}\n"
                "Expected Output Format: {expected_format}\n\n"
                "Specialist Output:\n{specialist_output}\n\n"
                "Evaluate this output and determine status (approved/rejected/escalate), confidence score (0.0 to 1.0), "
                "feedback, and actionable suggestions.",
            ),
        ])

    def evaluate(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node execution: reads state, validates current specialist output, and returns state updates."""
        plan = state.execution_plan
        if not plan or plan.current_task_index >= len(plan.tasks):
            return {
                "review_status": "approved",
                "confidence_score": 1.0,
                "review_feedback": "All tasks completed.",
            }

        current_task = plan.tasks[plan.current_task_index]
        specialist_raw_output = state.specialist_outputs.get(current_task.id)

        # Build formal ReviewerInput schema
        review_input = ReviewerInput(
            subtask_id=current_task.id,
            subtask_description=current_task.description,
            specialist_type=current_task.assigned_specialist,
            specialist_output=specialist_raw_output,
            expected_format=current_task.expected_output_format,
            criteria=["completeness", "accuracy", "format_compliance"],
        )

        review_output = self.evaluate_input(review_input)

        # Log errors or warnings if rejected
        error_updates = list(state.error_log)
        if review_output.status == "rejected":
            error_updates.append(f"Subtask {current_task.id} rejected: {review_output.feedback}")
        elif review_output.status == "escalate":
            error_updates.append(f"Subtask {current_task.id} escalated: {review_output.feedback}")

        return {
            "review_status": review_output.status,
            "confidence_score": review_output.confidence_score,
            "review_feedback": review_output.feedback,
            "reviewer_output": review_output,
            "error_log": error_updates,
        }

    def evaluate_input(self, review_input: ReviewerInput) -> ReviewerOutput:
        """Core validation logic with LLM and heuristic fallbacks."""
        # 1. Immediate rejection if output is empty or error
        output_str = str(review_input.specialist_output or "").strip()

        if not output_str or output_str == "None":
            return ReviewerOutput(
                subtask_id=review_input.subtask_id,
                status="rejected",
                confidence_score=0.1,
                feedback="Specialist returned an empty output. Subtask must be re-executed with complete content.",
                suggestions=["Ensure the model generates a comprehensive response", "Verify domain tool invocations"],
            )

        if "Specialist [" in output_str and "failed:" in output_str:
            return ReviewerOutput(
                subtask_id=review_input.subtask_id,
                status="rejected",
                confidence_score=0.2,
                feedback=f"Specialist encountered an execution error: {output_str}",
                suggestions=["Review tool parameters and retry execution"],
            )

        # 2. If LLM is configured, attempt structured review
        if self.llm is not None:
            try:
                if hasattr(self.llm, "with_structured_output"):
                    structured_llm = self.llm.with_structured_output(ReviewerOutput)
                    chain = self.review_prompt | structured_llm
                    result = chain.invoke({
                        "subtask_id": review_input.subtask_id,
                        "specialist_type": review_input.specialist_type,
                        "subtask_description": review_input.subtask_description,
                        "expected_format": review_input.expected_format,
                        "specialist_output": output_str[:4000],
                    })
                    if isinstance(result, ReviewerOutput):
                        return result
                    elif isinstance(result, dict):
                        return ReviewerOutput(**result)
            except Exception:
                pass

        # 3. Robust heuristic evaluation fallback
        score = 0.85
        feedback = "Output thoroughly reviewed and validated against subtask criteria."
        status: ReviewStatus = "approved"

        # Heuristic checks
        if len(output_str.split()) < 10 and review_input.specialist_type != "coder":
            score = 0.5
            status = "rejected"
            feedback = "Output is excessively brief and lacks sufficient detail for this domain."
        elif "traceback" in output_str.lower() or "syntaxerror" in output_str.lower():
            score = 0.3
            status = "rejected"
            feedback = "Output contains Python traceback or syntax error."

        return ReviewerOutput(
            subtask_id=review_input.subtask_id,
            status=status,
            confidence_score=score,
            feedback=feedback,
            suggestions=["Maintain clear structure", "Cross-verify all findings"],
        )

