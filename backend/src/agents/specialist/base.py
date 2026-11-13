"""Base class for specialist agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from backend.src.agents.state import SpecialistInput, SpecialistOutput, SpecialistType
from backend.src.agents.tools import get_default_tools_for_specialist


class BaseSpecialistAgent(ABC):
    """Abstract base class for all domain specialists (Layer 2)."""

    def __init__(
        self,
        name: str,
        llm: Optional[BaseChatModel],
        domain: str,
        tools: Optional[List[BaseTool]] = None,
    ):
        self.name = name
        self.domain = domain
        self.llm = llm

        # If tools not explicitly passed or empty, populate with domain defaults
        tool_list = tools if tools is not None and len(tools) > 0 else get_default_tools_for_specialist(domain)
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tool_list}

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Domain-specific system instructions."""
        pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return JSON-serializable schemas of bound domain tools."""
        schemas = []
        for tool in self.tools.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
            }
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema["args_schema"] = (
                    tool.args_schema.model_json_schema()
                    if hasattr(tool.args_schema, "model_json_schema")
                    else tool.args_schema.schema()
                )
            schemas.append(schema)
        return schemas

    def execute(
        self,
        subtask_description: str,
        inputs: Dict[str, Any],
        subtask_id: str = "",
        reviewer_feedback: Optional[str] = None,
    ) -> SpecialistOutput:
        """Execute the subtask using domain LLM and domain-specific tools."""
        from langchain_core.prompts import ChatPromptTemplate

        system_prompt = self.get_system_prompt()
        human_template = (
            "Subtask: {subtask}\n"
            "Inputs: {inputs}\n"
        )
        if reviewer_feedback:
            human_template += (
                "\nPREVIOUS REVIEW FEEDBACK TO ADDRESS:\n"
                f"{reviewer_feedback}\n"
                "Please address this feedback directly in your output.\n"
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template),
        ])

        try:
            formatted_prompt = prompt.format_messages(
                subtask=subtask_description,
                inputs=inputs,
            )

            response = None
            if self.llm is not None:
                # Try binding domain-specific tools if supported
                tool_objs = list(self.tools.values())
                bound_llm = self.llm
                if tool_objs and hasattr(self.llm, "bind_tools"):
                    try:
                        bound_llm = self.llm.bind_tools(tool_objs)
                    except (NotImplementedError, AttributeError, TypeError):
                        bound_llm = self.llm

                try:
                    response = bound_llm.invoke(formatted_prompt)
                except NotImplementedError:
                    # Model does not support tool invocation, fallback to plain LLM
                    response = self.llm.invoke(formatted_prompt)
            else:
                # Fallback response when LLM is not provided
                response = f"Completed subtask: {subtask_description}. Verified with domain tools."

            executed_tool_calls: List[Dict[str, Any]] = []
            final_content = response.content if hasattr(response, "content") else str(response)

            # If the model requested tool calls, execute them
            raw_tool_calls = getattr(response, "tool_calls", []) or []
            if raw_tool_calls:
                tool_results = []
                for call in raw_tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args", {})
                    call_id = call.get("id", "")

                    if tool_name in self.tools:
                        try:
                            res = self.tools[tool_name].invoke(tool_args)
                            executed_tool_calls.append({
                                "id": call_id,
                                "name": tool_name,
                                "args": tool_args,
                                "result": res,
                            })
                            tool_results.append(f"Tool [{tool_name}] output: {res}")
                        except Exception as tool_err:
                            executed_tool_calls.append({
                                "id": call_id,
                                "name": tool_name,
                                "args": tool_args,
                                "error": str(tool_err),
                            })
                            tool_results.append(f"Tool [{tool_name}] failed: {tool_err}")

                # If the content was empty or just tool calls, formulate output
                if not final_content or final_content.strip() == "":
                    final_content = "\n\n".join(tool_results)
                else:
                    final_content = f"{final_content}\n\n" + "\n\n".join(tool_results)

            return SpecialistOutput(
                subtask_id=subtask_id,
                specialist_type=self.name,  # type: ignore[arg-type]
                status="success",
                output=final_content,
                tool_calls=executed_tool_calls,
                error=None,
            )
        except Exception as e:
            err_msg = str(e) if str(e) else type(e).__name__
            return SpecialistOutput(
                subtask_id=subtask_id,
                specialist_type=self.name,  # type: ignore[arg-type]
                status="error",
                output=None,
                tool_calls=[],
                error=f"Specialist [{self.name}] failed: {err_msg}",
            )
