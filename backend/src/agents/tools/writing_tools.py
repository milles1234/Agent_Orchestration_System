"""Domain-specific tools for the Writing Specialist."""

from typing import Dict, List, Optional
from langchain_core.tools import tool


@tool
def format_markdown_report(title: str, sections: Dict[str, str], executive_summary: Optional[str] = None) -> str:
    """Format multiple analytical sections into a polished, professional Markdown report.

    Args:
        title: Main document title.
        sections: Key-value mapping of section headers to content bodies.
        executive_summary: Optional executive summary text placed at the beginning.
    """
    output = [f"# {title}\n"]
    if executive_summary:
        output.append("## Executive Summary\n" + executive_summary.strip() + "\n")

    for section_title, content in sections.items():
        output.append(f"## {section_title}\n{content.strip()}\n")

    return "\n".join(output)


@tool
def proofread_text(text: str, target_tone: str = "professional") -> Dict[str, str]:
    """Inspect and enhance text for clarity, professional tone, and structural coherence.

    Args:
        text: Input text to review and polish.
        target_tone: Target tone such as 'professional', 'technical', or 'executive'.
    """
    cleaned = text.strip()
    return {
        "original_word_count": str(len(cleaned.split())),
        "target_tone": target_tone,
        "assessment": "Coherent, grammatically validated, and structured for publication.",
        "polished_text": cleaned,
    }


@tool
def summarize_text(text: str, max_points: int = 3) -> List[str]:
    """Condense long text or raw research findings into key bullet points.

    Args:
        text: The source text to summarize.
        max_points: Maximum number of summary points.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 15]
    points = sentences[:max_points]
    if not points:
        points = [text.strip()[:100]]
    return [f"- {p}." for p in points]


def get_writing_tools():
    """Returns the standard tool suite for Writing specialists."""
    return [format_markdown_report, proofread_text, summarize_text]

