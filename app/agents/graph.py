"""The assistant graph: specialised agents wired together with LangGraph.

    START
      │
    input_guard ──(blocked)──────────────────────────────────────────┐
      │                                                               │
    memory_load                                                       │
      │                                                               │
    supervisor ──► next_step ─┬─► retrieval ────┐                     │
                              ├─► research ─────┤ (RLM subgraph)      │
                              ├─► tool_planner ─► tool_executor ──┐   │
                              │      (interrupt for approval)     │   │
                              │◄──────────── back to next_step ◄──┘   │
                              └─► response ◄── retry once ──┐         │
                                    │                       │         │
                                  validator ────────────────┘         │
                                    │                                 │
                                  memory_save ◄───────────────────────┘
                                    │
                                   END

The supervisor writes an ordered plan; `next_step` walks it, so the control flow is data
(visible in the state, the trace and the UI), not hidden inside a prompt.

Persistence: the checkpointer (AsyncSqliteSaver) stores state per thread
(thread_id = session id). That gives multi-turn memory and makes the human-approval
interrupt resumable from a separate HTTP request.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents import nodes
from app.agents.state import AssistantState

WORKERS = ["retrieval", "research", "tool_planner", "response"]


def build_graph(checkpointer=None):
    g = StateGraph(AssistantState)
    g.add_node("input_guard", nodes.input_guard)
    g.add_node("memory_load", nodes.memory_load)
    g.add_node("supervisor", nodes.supervisor)
    g.add_node("retrieval", nodes.retrieval)
    g.add_node("research", nodes.research)
    g.add_node("tool_planner", nodes.tool_planner)
    g.add_node("tool_executor", nodes.tool_executor)
    g.add_node("response", nodes.response)
    g.add_node("validator", nodes.validator)
    g.add_node("memory_save", nodes.memory_save)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", nodes.after_input_guard, ["memory_load", "memory_save"])
    g.add_edge("memory_load", "supervisor")
    g.add_conditional_edges("supervisor", nodes.next_step, WORKERS)
    g.add_conditional_edges("retrieval", nodes.next_step, WORKERS)
    g.add_conditional_edges("research", nodes.next_step, WORKERS)
    g.add_edge("tool_planner", "tool_executor")
    g.add_conditional_edges("tool_executor", nodes.next_step, WORKERS)
    g.add_edge("response", "validator")
    g.add_conditional_edges("validator", nodes.after_validator, ["response", "memory_save"])
    g.add_edge("memory_save", END)
    return g.compile(checkpointer=checkpointer, name="commercial_bank_assistant")
