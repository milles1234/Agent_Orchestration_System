"""Unit and integration tests for the three-layer agent orchestration system."""

import pytest
from typing import Any, Dict, List
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

from backend.src.agents.state import (
    AgentState,
    ExecutionPlan,
    SubTask,
    SupervisorInput,
    SupervisorOutput,
    SpecialistInput,
    SpecialistOutput,
    ReviewerInput,
    ReviewerOutput,
)
from backend.src.agents.tools import (
    get_default_tools_for_specialist,
    web_search,
    fetch_document,
    cite_source,
    calculate_statistics,
    query_dataset,
    aggregate_data,
    format_markdown_report,
    proofread_text,
    summarize_text,
    execute_python_code,
    validate_syntax,
)
from backend.src.agents.supervisor.planner import TaskDecompositionEngine
from backend.src.agents.specialist.researcher import ResearchSpecialist
from backend.src.agents.specialist.data_analyst import DataAnalystSpecialist
from backend.src.agents.specialist.writer import WriterSpecialist
from backend.src.agents.specialist.coder import CoderSpecialist
from backend.src.agents.reviewer.evaluator import ReviewerEvaluator
from backend.src.agents.graph import build_orchestrator_graph


class DummyMockLLM(BaseChatModel):
    """Deterministic mock LLM for testing without network/API dependencies."""

    custom_response: str = "Mock response providing comprehensive domain analysis and conclusions."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.custom_response))])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "dummy-mock"



# ============================================================================
# 1. Tests for Domain-Specific Tools
# ============================================================================

def test_research_tools():
    """Verify research specialist tools."""
    results = web_search.invoke({"query": "AI Agents in Enterprise"})
    assert len(results) > 0
    assert "Research Report" in results[0]["title"]

    doc = fetch_document.invoke({"document_id_or_url": "doc_101"})
    assert doc["id"] == "doc_101"
    assert doc["status"] == "retrieved"

    citation = cite_source.invoke({"title": "Multi-Agent Architectures", "author": "Smith", "year": 2025})
    assert citation == "[Smith (2025). Multi-Agent Architectures]"


def test_data_analysis_tools():
    """Verify data analyst specialist tools."""
    stats = calculate_statistics.invoke({"data": [10.0, 20.0, 30.0, 40.0, 50.0]})
    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0

    records = [
        {"dept": "Engineering", "spend": 100},
        {"dept": "Marketing", "spend": 50},
        {"dept": "Engineering", "spend": 150},
    ]
    filtered = query_dataset.invoke({"records": records, "filter_key": "dept", "filter_value": "Engineering"})
    assert len(filtered) == 2

    agg = aggregate_data.invoke({"records": records, "group_by": "dept", "agg_key": "spend"})
    assert agg["Engineering"] == 250.0
    assert agg["Marketing"] == 50.0


def test_writing_tools():
    """Verify writing specialist tools."""
    report = format_markdown_report.invoke({
        "title": "Quarterly Review",
        "sections": {"Key Findings": "Revenue increased by 15%."},
        "executive_summary": "Strong quarterly performance across all divisions.",
    })
    assert "# Quarterly Review" in report
    assert "## Executive Summary" in report
    assert "## Key Findings" in report

    proof = proofread_text.invoke({"text": "System architecture is solid.", "target_tone": "executive"})
    assert proof["target_tone"] == "executive"

    summary = summarize_text.invoke({"text": "First sentence is informative. Second sentence explains details. Third wraps up."})
    assert len(summary) >= 1


def test_code_execution_tools():
    """Verify code execution specialist tools."""
    code = "x = 40 + 2\nprint('Computed answer:', x)"
    result = execute_python_code.invoke({"code": code})
    assert result["success"] is True
    assert "Computed answer: 42" in result["stdout"]
    assert result["variables"].get("x") == "42"

    syntax_valid = validate_syntax.invoke({"code": "def hello():\n    return 'world'"})
    assert syntax_valid["valid"] is True

    syntax_invalid = validate_syntax.invoke({"code": "def broken_func(: pass"})
    assert syntax_invalid["valid"] is False


def test_default_tool_assignment():
    """Verify that domain default tools are properly registered."""
    research_tools = get_default_tools_for_specialist("researcher")
    data_tools = get_default_tools_for_specialist("analyst")
    writing_tools = get_default_tools_for_specialist("writer")
    coding_tools = get_default_tools_for_specialist("coder")

    assert len(research_tools) >= 2
    assert len(data_tools) >= 2
    assert len(writing_tools) >= 2
    assert len(coding_tools) >= 2


# ============================================================================
# 2. Tests for Layer 1: Supervisor Agent
# ============================================================================

def test_supervisor_planning_and_synthesis():
    """Verify supervisor decomposition, plan generation, and synthesis."""
    supervisor = TaskDecompositionEngine(llm=None)
    state = AgentState(
        task_id="task_super_1",
        user_request="Research renewable energy trends, analyze cost data, implement simulation code, and write an executive report.",
    )

    output: SupervisorOutput = supervisor.plan_task(state)
    assert isinstance(output, SupervisorOutput)
    plan = output.execution_plan
    assert len(plan.tasks) >= 3

    assigned_specialists = [t.assigned_specialist for t in plan.tasks]
    assert "researcher" in assigned_specialists
    assert "analyst" in assigned_specialists
    assert "coder" in assigned_specialists
    assert "writer" in assigned_specialists

    # Verify all required subtask fields are present
    for subtask in plan.tasks:
        assert subtask.description
        assert subtask.assigned_specialist in ("researcher", "analyst", "coder", "writer")
        assert isinstance(subtask.required_inputs, list)
        assert subtask.expected_output_format
        assert subtask.estimated_complexity in ("low", "medium", "high")
        assert subtask.expected_complexity in ("low", "medium", "high")

    # Test synthesis
    state.specialist_outputs = {
        "subtask_1": "Solar efficiency has grown by 24%.",
        "subtask_2": "Cost per kWh dropped from $0.12 to $0.05.",
    }
    synth = supervisor.synthesize_final_output(state)
    assert "Final Deliverable" in synth.final_output
    assert "Solar efficiency" in synth.final_output


def test_task_decomposition_structured_output_mock():
    """Verify decomposition with structured output LLM."""
    plan_to_return = ExecutionPlan(
        tasks=[
            SubTask(
                id="subtask_1",
                description="Gather quantum computing benchmark papers from 2024 to 2026",
                assigned_specialist="researcher",
                required_inputs=[],
                expected_output_format="markdown",
                estimated_complexity="medium",
            ),
            SubTask(
                id="subtask_2",
                description="Extract qubit error rates and calculate average gate fidelities",
                assigned_specialist="analyst",
                required_inputs=["subtask_1"],
                expected_output_format="json",
                estimated_complexity="high",
            ),
            SubTask(
                id="subtask_3",
                description="Simulate noise-free circuit execution on 12 qubits",
                assigned_specialist="coder",
                required_inputs=["subtask_2"],
                expected_output_format="python_code",
                estimated_complexity="high",
            ),
            SubTask(
                id="subtask_4",
                description="Write executive whitepaper summarizing findings",
                assigned_specialist="writer",
                required_inputs=["subtask_1", "subtask_2", "subtask_3"],
                expected_output_format="markdown",
                estimated_complexity="medium",
            ),
        ],
        current_task_index=0,
        reasoning="Multi-step sequential workflow from literature to benchmarking and documentation.",
    )

    class StructuredMockLLM(DummyMockLLM):
        def with_structured_output(self, schema, **kwargs):
            class StructuredRunner:
                def __init__(self, plan):
                    self.plan = plan
                def invoke(self, inputs):
                    return self.plan
            return StructuredRunner(plan_to_return)

    engine = TaskDecompositionEngine(llm=StructuredMockLLM())
    result_plan = engine.decompose("Benchmark quantum algorithms and write whitepaper")

    assert len(result_plan.tasks) == 4
    assert result_plan.tasks[0].assigned_specialist == "researcher"
    assert result_plan.tasks[1].assigned_specialist == "analyst"
    assert result_plan.tasks[2].assigned_specialist == "coder"
    assert result_plan.tasks[3].assigned_specialist == "writer"
    assert result_plan.tasks[1].required_inputs == ["subtask_1"]
    assert result_plan.tasks[1].estimated_complexity == "high"
    assert result_plan.tasks[1].expected_output_format == "json"

    # Display formatting
    table = engine.format_plan_for_display(result_plan)
    assert "| Step | Subtask ID | Specialist |" in table
    assert "subtask_1" in table
    assert "quantum" in table.lower()


def test_task_decomposition_json_parser_fallback():
    """Verify decomposition when LLM outputs raw JSON code fence."""
    raw_json = """
    ```json
    {
        "tasks": [
            {
                "id": "subtask_1",
                "description": "Fetch financial reports",
                "assigned_specialist": "researcher",
                "required_inputs": [],
                "expected_output_format": "text",
                "estimated_complexity": "low"
            },
            {
                "id": "subtask_2",
                "description": "Calculate quarterly YoY revenue margins",
                "assigned_specialist": "analyst",
                "required_inputs": ["subtask_1"],
                "expected_output_format": "csv",
                "estimated_complexity": "medium"
            }
        ],
        "current_task_index": 0,
        "reasoning": "Data ingestion followed by financial analytics."
    }
    ```
    """
    mock_llm = DummyMockLLM(custom_response=raw_json)
    # Ensure with_structured_output is not present or bypassed
    engine = TaskDecompositionEngine(llm=mock_llm)
    plan = engine.decompose("Analyze quarterly revenue trends")

    assert len(plan.tasks) == 2
    assert plan.tasks[0].assigned_specialist == "researcher"
    assert plan.tasks[1].assigned_specialist == "analyst"
    assert plan.tasks[1].estimated_complexity == "medium"
    assert plan.tasks[1].expected_output_format == "csv"
    assert plan.tasks[1].required_inputs == ["subtask_1"]



# ============================================================================
# 3. Tests for Layer 2: Specialist Agents
# ============================================================================

def test_specialists_execution_and_schema():
    """Verify each specialist executes and outputs according to SpecialistOutput schema."""
    mock_llm = DummyMockLLM()

    # Researcher
    researcher = ResearchSpecialist(llm=mock_llm)
    res_out: SpecialistOutput = researcher.execute(
        subtask_description="Research market dynamics",
        inputs={},
        subtask_id="sub_res_1",
    )
    assert isinstance(res_out, SpecialistOutput)
    assert res_out.status == "success"
    assert res_out.specialist_type == "researcher"
    assert res_out.output is not None

    # Analyst
    analyst = DataAnalystSpecialist(llm=mock_llm)
    ana_out: SpecialistOutput = analyst.execute(
        subtask_description="Analyze revenue metrics",
        inputs={"data": [10, 20, 30]},
        subtask_id="sub_ana_1",
    )
    assert isinstance(ana_out, SpecialistOutput)
    assert ana_out.status == "success"
    assert ana_out.specialist_type == "analyst"

    # Coder
    coder = CoderSpecialist(llm=mock_llm)
    cod_out: SpecialistOutput = coder.execute(
        subtask_description="Implement quicksort algorithm",
        inputs={},
        subtask_id="sub_cod_1",
    )
    assert isinstance(cod_out, SpecialistOutput)
    assert cod_out.status == "success"
    assert cod_out.specialist_type == "coder"

    # Writer
    writer = WriterSpecialist(llm=mock_llm)
    wri_out: SpecialistOutput = writer.execute(
        subtask_description="Draft summary report",
        inputs={"research": "Findings"},
        subtask_id="sub_wri_1",
    )
    assert isinstance(wri_out, SpecialistOutput)
    assert wri_out.status == "success"
    assert wri_out.specialist_type == "writer"


# ============================================================================
# 4. Tests for Layer 3: Reviewer Agent
# ============================================================================

def test_reviewer_evaluation_rules():
    """Verify reviewer approval, rejection, and schema validation."""
    reviewer = ReviewerEvaluator(llm=None)

    # Valid output should be approved
    valid_input = ReviewerInput(
        subtask_id="task_1",
        subtask_description="Research renewable trends",
        specialist_type="researcher",
        specialist_output="Comprehensive analysis of global renewable adoption showing 24% CAGR over five years.",
        expected_format="text",
    )
    res: ReviewerOutput = reviewer.evaluate_input(valid_input)
    assert res.status == "approved"
    assert res.confidence_score >= 0.7

    # Empty output should be rejected
    empty_input = ReviewerInput(
        subtask_id="task_2",
        subtask_description="Research renewable trends",
        specialist_type="researcher",
        specialist_output="",
        expected_format="text",
    )
    res_empty: ReviewerOutput = reviewer.evaluate_input(empty_input)
    assert res_empty.status == "rejected"

    # Error output should be rejected
    error_input = ReviewerInput(
        subtask_id="task_3",
        subtask_description="Code execution",
        specialist_type="coder",
        specialist_output="Specialist [coder] failed: syntax error in script",
        expected_format="code",
    )
    res_err: ReviewerOutput = reviewer.evaluate_input(error_input)
    assert res_err.status == "rejected"


# ============================================================================
# 5. Integration Test: Full Graph Execution
# ============================================================================

def test_end_to_end_graph_execution():
    """Verify that the LangGraph workflow runs end-to-end across all three layers."""
    mock_llm = DummyMockLLM(
        custom_response="Detailed findings and technical specifications thoroughly verified against criteria."
    )
    app = build_orchestrator_graph(
        supervisor_llm=mock_llm,
        specialist_llm=mock_llm,
        reviewer_llm=mock_llm,
    )

    initial_state = AgentState(
        task_id="integration_task_1",
        user_request="Research solar battery storage, analyze market efficiency, and write a summary.",
    )

    result = app.invoke(initial_state)

    assert result["final_output"] is not None
    assert len(result["specialist_outputs"]) > 0
    assert result["review_status"] == "approved"
    assert result["execution_plan"] is not None
    assert result["execution_plan"].current_task_index == len(result["execution_plan"].tasks)
