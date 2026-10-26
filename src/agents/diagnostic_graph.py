from langgraph.graph import END, START, StateGraph

from agents.context import AgentContext
from agents.diagnostic_nodes import (
    analyze_evidence,
    execute_tools,
    finish_analysis,
    finish_direct,
    finish_failure,
    prepare_evidence,
    request_model,
    route_after_analysis,
    route_after_evidence,
    route_after_model,
    route_after_tools,
)
from agents.diagnostic_state import DiagnosticState, start_diagnosis

builder = StateGraph(
    DiagnosticState,
    context_schema=AgentContext,
)
builder.add_node("start_diagnosis", start_diagnosis)
builder.add_node("request_model", request_model)
builder.add_node("execute_tools", execute_tools)
builder.add_node("prepare_evidence", prepare_evidence)
builder.add_node("analyze_evidence", analyze_evidence)
builder.add_node("finish_analysis", finish_analysis)
builder.add_node("finish_direct", finish_direct)
builder.add_node("finish_failure", finish_failure)

builder.add_edge(START, "start_diagnosis")
builder.add_edge("start_diagnosis", "request_model")

builder.add_edge("finish_direct", END)
builder.add_edge("finish_analysis", END)
builder.add_edge("finish_failure", END)

builder.add_conditional_edges(
    "request_model",
    route_after_model,
    {
        "tools": "execute_tools",
        "finish": "finish_direct",
    },
)

builder.add_conditional_edges(
    "execute_tools",
    route_after_tools,
    {
        "prepare_evidence": "prepare_evidence",
        "failure": "finish_failure",
    },
)

builder.add_conditional_edges(
    "prepare_evidence",
    route_after_evidence,
    {
        "analyze": "analyze_evidence",
        "failure": "finish_failure",
    },
)

builder.add_conditional_edges(
    "analyze_evidence",
    route_after_analysis,
    {
        "finish": "finish_analysis",
        "failure": "finish_failure",
    },
)

diagnostic_graph = builder.compile()
