from backend.src.agents.specialist.base import BaseSpecialistAgent
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any


class CoderSpecialist(BaseSpecialistAgent):
    def execute(self, subtask_description: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Software Engineer and Code Execution Specialist. Write clean, bug-free code and use available sandbox or file tools to accomplish the task."),
            ("human", "Subtask: {subtask}\nInputs: {inputs}")
        ])

        bound_llm = self.llm.bind_tools(list(self.tools.values()))

        try:
            response = bound_llm.invoke(prompt.format(subtask=subtask_description, inputs=inputs))
            return {
                "status": "success",
                "output": response.content,
                "tool_calls": getattr(response, "tool_calls", [])
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }