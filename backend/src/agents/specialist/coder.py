"""Code Execution Specialist Agent."""

from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from backend.src.agents.specialist.base import BaseSpecialistAgent


class CoderSpecialist(BaseSpecialistAgent):
    """Specialist responsible for code writing, syntax verification, and code execution."""

    def __init__(
        self,
        name: str = "coder",
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        super().__init__(name=name, llm=llm, domain="code execution", tools=tools)

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Software Engineer and Code Execution Specialist Agent. Your mission is to "
            "write clean, modular, and bug-free code, perform static syntax validation, and execute code snippets "
            "using your execution sandbox tools. Always inspect runtime outputs and handle potential errors proactively."
        )


# Backward compatibility alias
CodeExecutionSpecialist = CoderSpecialist
