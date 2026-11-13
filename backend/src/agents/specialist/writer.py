"""Writing Specialist Agent."""

from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from backend.src.agents.specialist.base import BaseSpecialistAgent


class WriterSpecialist(BaseSpecialistAgent):
    """Specialist responsible for synthesis, documentation, drafting reports, and executive communication."""

    def __init__(
        self,
        name: str = "writer",
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        super().__init__(name=name, llm=llm, domain="writing", tools=tools)

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Technical Writer and Communications Specialist Agent. Your mission is to "
            "synthesize raw research, statistical insights, and code outputs into clear, compelling, and "
            "rigorously formatted deliverables. Use your Markdown reporting, text summarization, and proofreading "
            "tools to produce executive-ready documents."
        )
