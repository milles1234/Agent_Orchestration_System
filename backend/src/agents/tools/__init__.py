"""Domain-specific tools package for specialist agents."""

from typing import List
from langchain_core.tools import BaseTool

from backend.src.agents.tools.research_tools import get_research_tools, web_search, fetch_document, cite_source
from backend.src.agents.tools.data_tools import get_data_tools, calculate_statistics, query_dataset, aggregate_data
from backend.src.agents.tools.writing_tools import get_writing_tools, format_markdown_report, proofread_text, summarize_text
from backend.src.agents.tools.coding_tools import get_coding_tools, execute_python_code, validate_syntax


def get_default_tools_for_specialist(specialist_domain: str) -> List[BaseTool]:
    """Provide standard domain-specific tools based on specialist domain."""
    normalized = specialist_domain.lower().strip()
    if normalized in ("researcher", "research"):
        return get_research_tools()
    elif normalized in ("data_analyst", "analyst", "data analysis", "data"):
        return get_data_tools()
    elif normalized in ("writer", "writing"):
        return get_writing_tools()
    elif normalized in ("coder", "code execution", "code", "software_engineer"):
        return get_coding_tools()
    return []


__all__ = [
    "get_default_tools_for_specialist",
    "get_research_tools",
    "get_data_tools",
    "get_writing_tools",
    "get_coding_tools",
    "web_search",
    "fetch_document",
    "cite_source",
    "calculate_statistics",
    "query_dataset",
    "aggregate_data",
    "format_markdown_report",
    "proofread_text",
    "summarize_text",
    "execute_python_code",
    "validate_syntax",
]
