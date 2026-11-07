
from typing import Any
from abc import abstractmethod
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from abc import ABC
from typing import List, Dict

class BaseSpecialistAgent(ABC):
    def __init__(self, name: str, llm: BaseChatModel, tools: List[BaseTool]):
        self.name = name
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    @abstractmethod 
    def execute(self, substack_description: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass
        # But the parent class doesn't decide what that method actually does.
        # Because every agent performs a different task.



    def get_tool_schemas(self) -> List[dict[str, Any]]:
        return[
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema.schema() if tool.args_schema else None

            }
            for tool in self.tools.values()
        ]





         