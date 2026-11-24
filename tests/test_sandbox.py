import pytest

from app.errors import SandboxError
from app.security.sandbox import run_sandboxed


def test_allowed_code_runs():
    env = run_sandboxed("result = sorted(Counter(xs).items())", {"xs": ["a", "b", "a"]})
    assert env["result"] == [("a", 2), ("b", 1)]


@pytest.mark.parametrize("code", [
    "import os",
    "from os import path",
    "open('x')",
    "x = ().__class__",
    "eval('1')",
    "getattr(1, 'real')",
    "'{0.__class__}'.format(1)",
    "class A: pass",
    "global x",
])
def test_dangerous_code_is_rejected(code):
    with pytest.raises(SandboxError):
        run_sandboxed(code, {})


def test_infinite_loop_is_stopped():
    with pytest.raises(SandboxError, match="step limit"):
        run_sandboxed("while True:\n    pass", {})


def test_runtime_errors_are_wrapped():
    with pytest.raises(SandboxError, match="ZeroDivisionError"):
        run_sandboxed("x = 1 / 0", {})
