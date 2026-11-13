"""Domain-specific tools for the Code Execution Specialist."""

import ast
import io
import sys
from typing import Any, Dict
from langchain_core.tools import tool


@tool
def execute_python_code(code: str) -> Dict[str, Any]:
    """Execute a Python code snippet in an isolated namespace and capture stdout and return variables.

    Args:
        code: Python source code string to execute.
    """
    stdout_buffer = io.StringIO()
    local_env: Dict[str, Any] = {}
    old_stdout = sys.stdout

    try:
        sys.stdout = stdout_buffer
        exec(code, {"__builtins__": __builtins__}, local_env)
        stdout_output = stdout_buffer.getvalue()
        # Filter out builtins from returned variables
        result_vars = {k: str(v) for k, v in local_env.items() if not k.startswith("__")}
        return {
            "success": True,
            "stdout": stdout_output,
            "variables": result_vars,
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "stdout": stdout_buffer.getvalue(),
            "variables": {},
            "error": f"{type(exc).__name__}: {str(exc)}",
        }
    finally:
        sys.stdout = old_stdout


@tool
def validate_syntax(code: str) -> Dict[str, Any]:
    """Perform static syntax analysis on Python code using the AST parser without executing it.

    Args:
        code: Python code string to inspect.
    """
    try:
        tree = ast.parse(code)
        num_statements = len(tree.body)
        return {
            "valid": True,
            "statements_count": num_statements,
            "message": "Syntax is valid and parsed successfully.",
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}",
        }


def get_coding_tools():
    """Returns the standard tool suite for Code Execution specialists."""
    return [execute_python_code, validate_syntax]

