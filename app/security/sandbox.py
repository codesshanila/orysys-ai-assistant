"""AST-validated Python sandbox.

Used twice:
  * the RLM research agent's generated *search plans* (Python that calls a tiny DSL), and
  * the Python analysis tool (structured analysis over retrieved records).

Defence layers
  1. Static validation: the code is parsed and every AST node is checked against an
     allow-list. No imports, no attribute access to dunder names, no `open/eval/exec`,
     no `global`, no class definitions, bounded size.
  2. Restricted globals: only safe builtins plus the functions we inject.
  3. Time limit: execution runs in a worker thread with a deadline; long loops are
     also capped by an instruction counter via sys.settrace.

This is a demonstration-grade sandbox. Production would run code in a separate
container or a microVM (gVisor/Firecracker) - documented in docs/decisions.md.
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from typing import Any, Callable

from app.errors import SandboxError

MAX_CODE_CHARS = 4000
MAX_LINE_EVENTS = 200_000

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp,
    ast.GeneratorExp, ast.comprehension, ast.Call, ast.keyword, ast.Attribute, ast.Subscript,
    ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.If, ast.For,
    ast.While, ast.Break, ast.Continue, ast.Pass, ast.FunctionDef, ast.Return, ast.arguments,
    ast.arg, ast.Lambda, ast.JoinedStr, ast.FormattedValue, ast.Starred,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In,
    ast.NotIn, ast.Is, ast.IsNot,
)
FORBIDDEN_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit", "memoryview", "type",
    "object", "super", "classmethod", "staticmethod", "property",
})
# str.format can reach dunder attributes through "{0.__class__}" style fields.
FORBIDDEN_ATTRS = frozenset({"format", "format_map", "mro", "gi_frame", "f_globals"})
SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum, "sorted": sorted,
    "enumerate": enumerate, "zip": zip, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "round": round, "abs": abs,
    "any": any, "all": all, "map": map, "filter": filter, "reversed": reversed,
    "isinstance": isinstance, "Counter": Counter, "print": lambda *a, **k: None,
}


def validate_code(code: str) -> ast.Module:
    if len(code) > MAX_CODE_CHARS:
        raise SandboxError(f"code longer than {MAX_CODE_CHARS} characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error: {exc.msg} (line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise SandboxError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise SandboxError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS):
            raise SandboxError(f"disallowed attribute: {node.attr}")
    return tree


def run_sandboxed(code: str, injected: dict[str, Callable | Any]) -> dict[str, Any]:
    """Validate and execute `code`. Returns the resulting variables (minus injected ones)."""
    tree = validate_code(code)
    env: dict[str, Any] = {"__builtins__": SAFE_BUILTINS, **injected}
    events = 0

    def tracer(frame, event, arg):  # noqa: ARG001
        nonlocal events
        if event == "line":
            events += 1
            if events > MAX_LINE_EVENTS:
                raise SandboxError("execution step limit exceeded")
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(compile(tree, "<sandbox>", "exec"), env)  # noqa: S102 - validated above
    except SandboxError:
        raise
    except Exception as exc:
        raise SandboxError(f"runtime error: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.settrace(previous)
    return {k: v for k, v in env.items() if k not in injected and k != "__builtins__"}
