"""Research Specialist Agent."""

from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from backend.src.agents.specialist.base import BaseSpecialistAgent


class ResearchSpecialist(BaseSpecialistAgent):
    """Specialist responsible for deep research, information gathering, and fact retrieval."""

    def __init__(
        self,
        name: str = "researcher",
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        super().__init__(name=name, llm=llm, domain="research", tools=tools)

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Research Specialist Agent. Your mission is to gather accurate, "
            "comprehensive, and verified factual information. Use your web search, document retrieval, "
            "and citation tools to discover domain evidence. Always cite your sources and provide structured, "
            "objective summaries of the findings."
        )
