


from backend.src.agents.state import SubTask
from langchain_core.prompts import ChatPromptTemplate
from ast import Dict, Any
from backend.src.agents.specialist.base import BaseSpecialistAgent

class ResearchSpecialist(BaseSpecialistAgent):
    def execute(self, substack_description: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Research Agent. Use avaliable tools to answer the user query accurately."),
            ("human", "Subtask: {subtask}\nInputs: {inputs}")
        ])
        

        # chain execuation logic with tools bound with self.llm

        bound_llm = self.llm.bind_tools(list(self.tools.values()))


        try: 
            response = bound_llm.invoke(prompt.format(subtask= substack_description, inputs=inputs))
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

        





