"""Data Analysis Specialist Agent."""

from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from backend.src.agents.specialist.base import BaseSpecialistAgent


class DataAnalystSpecialist(BaseSpecialistAgent):
    """Specialist responsible for data processing, statistical computations, and quantitative analysis."""

    def __init__(
        self,
        name: str = "analyst",
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
    ):
        super().__init__(name=name, llm=llm, domain="data analysis", tools=tools)

    def get_system_prompt(self) -> str:
        return (
            "You are an expert Data Analysis Specialist Agent. Your mission is to query structured datasets, "
            "perform statistical computations, calculate key performance metrics, and aggregate metrics. "
            "Use your data calculation, filtering, and aggregation tools to extract precise analytical insights "
            "and present them with clear metrics and tables."
        )


# Backward compatibility alias
AnalystSpecialist = DataAnalystSpecialist
