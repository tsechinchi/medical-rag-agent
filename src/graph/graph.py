from __future__ import annotations

from langgraph.graph import END, StateGraph

from config.config import FAITHFULNESS_THRESHOLD as THRESHOLD, MAX_RETRIES
from src.graph.nodes.critic import critic
from src.graph.nodes.generator import generator
from src.graph.nodes.planner import planner
from src.graph.nodes.retriever import retriever
from src.graph.nodes.synthesizer import synthesizer
from src.graph.state import AgentState


def _route_after_critic(state: AgentState) -> str:
    retries = state.get("retry_count", 0)
    faithfulness = state.get("faithfulness_score", 0.0)
    if faithfulness < THRESHOLD and retries < MAX_RETRIES:
        return "retry"
    return "synthesize"


def _increment_retry(state: AgentState) -> AgentState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("generator", generator)
    graph.add_node("critic", critic)
    graph.add_node("retry_counter", _increment_retry)
    graph.add_node("synthesizer", synthesizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "retry": "retry_counter",
            "synthesize": "synthesizer",
        },
    )
    graph.add_edge("retry_counter", "planner")
    graph.add_edge("synthesizer", END)
    return graph


def compile_graph():
    return build_graph().compile()
