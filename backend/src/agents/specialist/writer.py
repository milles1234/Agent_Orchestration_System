from backend.src.agents.specialist.base import BaseSpecialistAgent
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any


class WriterSpecialist(BaseSpecialistAgent):
    def execute(self, subtask_description: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Technical Writer and Communications Specialist. Synthesize raw notes, research, and code outputs into a polished, professional final document."),
            ("human", "Subtask: {subtask}\nInputs: {inputs}")
        ])

        # Writers typically format text, but can also bind tools if file writing or external publishing is needed
        bound_llm = self.llm.bind_tools(list(self.tools.values())) if self.tools else self.llm

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