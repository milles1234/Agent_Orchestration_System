"""Supervisor Agent (Layer 1): Core Task Decomposition Engine.

Receives complex tasks, breaks them down into small, ordered subtasks assigned to domain
specialists, and uses structured output to enforce a valid execution plan.
"""

import json
import re
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.src.agents.state import (
    AgentState,
    ExecutionPlan,
    SubTask,
    SupervisorInput,
    SupervisorOutput,
    SupervisorSynthesisInput,
    SupervisorSynthesisOutput,
)


SPECIALIST_SYNONYMS: Dict[str, str] = {
    "research": "researcher",
    "researcher": "researcher",
    "web_research": "researcher",
    "analyst": "analyst",
    "data_analyst": "analyst",
    "data": "analyst",
    "analytics": "analyst",
    "coder": "coder",
    "code": "coder",
    "code_execution": "coder",
    "software_engineer": "coder",
    "developer": "coder",
    "writer": "writer",
    "writing": "writer",
    "technical_writer": "writer",
    "documentation": "writer",
}


class TaskDecompositionEngine:
    """Supervisor Agent's core capability: Decomposes complex tasks into an ordered

    sequence of atomic subtasks assigned to specialists, enforcing a valid execution plan
    via structured output.
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=ExecutionPlan)

        format_instructions = self.parser.get_format_instructions()

        self.planner_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Supervisor Agent orchestrating a team of domain specialists:\n"
                "- researcher: Fact retrieval, web search, literature review, domain verification, and citations.\n"
                "- analyst: Data analysis, statistical computations, table querying, metrics aggregation.\n"
                "- coder: Python software implementation, script execution, syntax validation, algorithms.\n"
                "- writer: Document synthesis, executive summaries, Markdown formatting, technical communication.\n\n"
                "Your core capability is Task Decomposition. When given a complex user task, break it down into an "
                "ordered, sequential list of small, atomic subtasks.\n\n"
                "Every subtask must strictly specify:\n"
                "1. id: unique sequential ID (e.g., 'subtask_1', 'subtask_2')\n"
                "2. description: clear, actionable instructions for what the specialist must do\n"
                "3. assigned_specialist: one of 'researcher', 'analyst', 'coder', 'writer'\n"
                "4. required_inputs: list of prerequisite subtask IDs (e.g. ['subtask_1']) or input keys needed\n"
                "5. expected_output_format: required format (e.g., 'markdown', 'json', 'python_code', 'text', 'csv')\n"
                "6. estimated_complexity: difficulty rating ('low', 'medium', or 'high')\n\n"
                "Ensure logical dependency ordering: prerequisites must precede dependent steps.\n"
                "Provide a clear strategic 'reasoning' for the decomposition.\n\n"
                "{format_instructions}",
            ),
            (
                "human",
                "Task Request:\n{user_request}\n\n"
                "Retrieved Memories / Context:\n{retrieved_memories}\n\n"
                "Decompose this task into a structured ExecutionPlan.",
            ),
        ]).partial(format_instructions=format_instructions)

        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an executive Supervisor Agent. Your role is to synthesize the reviewed outputs from your "
                "specialist team into a comprehensive, high-quality final deliverable that fully satisfies the user request.",
            ),
            (
                "human",
                "User Request: {user_request}\n\n"
                "Specialist Deliverables:\n{specialist_deliverables}\n\n"
                "Provide a complete, cohesive, and professional final response.",
            ),
        ])

    def decompose(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        retrieved_memories: Optional[List[Dict[str, Any]]] = None,
    ) -> ExecutionPlan:
        """Decompose a complex task into an ordered ExecutionPlan using structured output

        with automatic fallback mechanisms.
        """
        memories_str = str(retrieved_memories or (context.get("memories") if context else []))
        plan: Optional[ExecutionPlan] = None

        if self.llm is not None:
            # 1. Primary approach: Native structured output if supported by LLM
            if hasattr(self.llm, "with_structured_output"):
                try:
                    structured_llm = self.llm.with_structured_output(ExecutionPlan)
                    planner_chain = self.planner_prompt | structured_llm
                    result = planner_chain.invoke({
                        "user_request": task,
                        "retrieved_memories": memories_str,
                    })
                    if isinstance(result, ExecutionPlan):
                        plan = result
                    elif isinstance(result, dict):
                        plan = ExecutionPlan.model_validate(result)
                except Exception:
                    plan = None

            # 2. Secondary approach: Pydantic parser via prompt format instructions
            if plan is None:
                try:
                    chain = self.planner_prompt | self.llm
                    response = chain.invoke({
                        "user_request": task,
                        "retrieved_memories": memories_str,
                    })
                    raw_content = response.content if hasattr(response, "content") else str(response)

                    # Extract JSON block if wrapped in markdown fences
                    cleaned = self._extract_json_block(raw_content)
                    parsed_dict = json.loads(cleaned)
                    plan = ExecutionPlan.model_validate(parsed_dict)
                except Exception:
                    plan = None

        # 3. Tertiary fallback: Rule-based heuristic decomposition
        if plan is None or not plan.tasks:
            plan = self._heuristic_decompose(task)

        # Enforce sanitization and validation on the execution plan
        return self._validate_and_sanitize_plan(plan, task)

    def plan_task(self, state: AgentState) -> SupervisorOutput:
        """LangGraph node method: takes AgentState, breaks user request into an ExecutionPlan,

        and returns SupervisorOutput.
        """
        # Validate Input Schema
        _ = SupervisorInput(
            task_id=state.task_id or "task_0",
            user_request=state.user_request,
            context={"memories": state.retrieved_memories},
        )

        plan = self.decompose(
            task=state.user_request,
            retrieved_memories=state.retrieved_memories,
        )

        return SupervisorOutput(
            execution_plan=plan,
            reasoning=plan.reasoning or "Task decomposed into sequential specialist stages.",
        )

    def synthesize_final_output(self, state: AgentState) -> SupervisorSynthesisOutput:
        """Synthesize all reviewed specialist outputs into the final deliverable."""
        deliverables_text = "\n\n".join(
            f"### Subtask [{k}]\n{v}" for k, v in state.specialist_outputs.items()
        )

        _ = SupervisorSynthesisInput(
            user_request=state.user_request,
            specialist_outputs=state.specialist_outputs,
        )

        if self.llm is not None and deliverables_text.strip():
            try:
                chain = self.synthesis_prompt | self.llm
                response = chain.invoke({
                    "user_request": state.user_request,
                    "specialist_deliverables": deliverables_text,
                })
                content = response.content if hasattr(response, "content") else str(response)
                return SupervisorSynthesisOutput(final_output=content)
            except Exception:
                pass

        # Fallback synthesis formatting
        fallback_output = (
            f"# Final Deliverable for: {state.user_request}\n\n"
            f"{deliverables_text}\n\n"
            "---\n*Synthesized by Supervisor Agent after specialist execution and reviewer validation.*"
        )
        return SupervisorSynthesisOutput(final_output=fallback_output)

    def format_plan_for_display(self, plan: ExecutionPlan) -> str:
        """Return a formatted Markdown table of the execution plan."""
        rows = [
            "| Step | Subtask ID | Specialist | Complexity | Output Format | Required Inputs | Description |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, task in enumerate(plan.tasks, start=1):
            inputs_str = ", ".join(task.required_inputs) if task.required_inputs else "None"
            desc = task.description.replace("\n", " ").strip()
            if len(desc) > 60:
                desc = desc[:57] + "..."
            complexity = task.estimated_complexity or task.expected_complexity or "medium"
            rows.append(
                f"| {idx} | `{task.id}` | **{task.assigned_specialist}** | {complexity} | "
                f"`{task.expected_output_format}` | {inputs_str} | {desc} |"
            )
        return "\n".join(rows)

    def _extract_json_block(self, text: str) -> str:
        """Extract a JSON object from text or markdown code blocks."""
        # Find json block inside ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1].strip()
        return text.strip()

    def _validate_and_sanitize_plan(self, plan: ExecutionPlan, original_task: str) -> ExecutionPlan:
        """Enforces schema compliance and field normalization across all subtasks."""
        sanitized_tasks: List[SubTask] = []

        for idx, task in enumerate(plan.tasks, start=1):
            task_id = task.id or f"subtask_{idx}"
            if not task_id.startswith("subtask_"):
                task_id = f"subtask_{idx}"

            # Normalize specialist
            raw_specialist = str(task.assigned_specialist).lower().strip()
            specialist = SPECIALIST_SYNONYMS.get(raw_specialist, "researcher")

            # Normalize complexity
            complexity = task.estimated_complexity or task.expected_complexity or "medium"
            if complexity not in ("low", "medium", "high"):
                complexity = "medium"

            # Normalize format
            fmt = task.expected_output_format or "text"

            # Normalize required inputs
            inputs = [str(inp) for inp in task.required_inputs if inp]

            sanitized_tasks.append(SubTask(
                id=task_id,
                description=task.description or f"Execute phase {idx} for: {original_task[:40]}",
                assigned_specialist=specialist,  # type: ignore[arg-type]
                required_inputs=inputs,
                expected_output_format=fmt,
                estimated_complexity=complexity,  # type: ignore[arg-type]
                expected_complexity=complexity,  # type: ignore[arg-type]
                status="pending",
                result=None,
                error=None,
            ))

        reasoning = plan.reasoning or (
            f"Structured {len(sanitized_tasks)}-stage decomposition ensuring logical handoffs "
            "across domain specialists."
        )

        return ExecutionPlan(
            tasks=sanitized_tasks,
            current_task_index=0,
            reasoning=reasoning,
        )

    def _heuristic_decompose(self, request: str) -> ExecutionPlan:
        """Heuristic fallback that systematically breaks down requests based on domain keywords."""
        req_lower = request.lower()
        tasks: List[SubTask] = []
        counter = 1

        # 1. Research phase: Information gathering, literature review, background discovery
        has_research_signals = any(w in req_lower for w in [
            "research", "find", "search", "explore", "investigate", "what is",
            "trends", "compare", "overview", "study", "market", "competitor",
        ])
        if has_research_signals or len(tasks) == 0:
            tasks.append(SubTask(
                id=f"subtask_{counter}",
                description=f"Conduct domain research and collect background knowledge on: {request}",
                assigned_specialist="researcher",
                required_inputs=[],
                expected_output_format="Structured research summary with source citations",
                estimated_complexity="medium",
                expected_complexity="medium",
                status="pending",
            ))
            counter += 1

        # 2. Data analysis phase: Statistical computations, quantitative evaluation, dataset processing
        has_data_signals = any(w in req_lower for w in [
            "data", "analyze", "statistics", "dataset", "metrics", "calculate",
            "numbers", "aggregate", "revenue", "costs", "percentages", "kpi",
        ])
        if has_data_signals:
            prev_input = [f"subtask_{counter-1}"] if counter > 1 else []
            tasks.append(SubTask(
                id=f"subtask_{counter}",
                description=f"Perform quantitative data analysis and compute metrics for: {request}",
                assigned_specialist="analyst",
                required_inputs=prev_input,
                expected_output_format="Statistical summary and metric table",
                estimated_complexity="medium",
                expected_complexity="medium",
                status="pending",
            ))
            counter += 1

        # 3. Code execution phase: Scripting, algorithms, Python execution, syntax checking
        has_code_signals = any(w in req_lower for w in [
            "code", "script", "program", "python", "algorithm", "implement",
            "execute", "benchmark", "function", "develop", "software",
        ])
        if has_code_signals:
            prev_input = [f"subtask_{counter-1}"] if counter > 1 else []
            tasks.append(SubTask(
                id=f"subtask_{counter}",
                description=f"Implement and execute code solution for: {request}",
                assigned_specialist="coder",
                required_inputs=prev_input,
                expected_output_format="Tested Python code and execution output",
                estimated_complexity="high",
                expected_complexity="high",
                status="pending",
            ))
            counter += 1

        # 4. Writing phase: Executive reporting, documentation, final publication formatting
        dep_inputs = [f"subtask_{i}" for i in range(1, counter)]
        tasks.append(SubTask(
            id=f"subtask_{counter}",
            description=f"Synthesize specialist deliverables into an executive report for: {request}",
            assigned_specialist="writer",
            required_inputs=dep_inputs,
            expected_output_format="Polished Markdown report",
            estimated_complexity="medium",
            expected_complexity="medium",
            status="pending",
        ))

        return ExecutionPlan(
            tasks=tasks,
            current_task_index=0,
            reasoning=f"Automatic multi-stage decomposition for '{request[:60]}...' with sequential handoffs.",
        )
