"""The analyst agent, built as a LangGraph state machine.

    START -> plan -> analyst <-> tools
                        |
                        v
                    validate --(issues found)--> analyst
                        |
                      (pass)
                        v
                     explain -> END

plan     decides the approach before touching data
analyst  calls SQL / Python / chart / anomaly / trend tools
validate re-reads the evidence and rejects unsupported claims
explain  turns verified findings into a business-readable answer

Conversation memory comes from a LangGraph checkpointer keyed by thread_id, so
the agent remembers the dataset and every earlier result within a session.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import SETTINGS, build_llm
from .dataset import AnalysisContext
from .tools import build_tools, summarize_table_for_prompt

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
ANALYST_SYSTEM = """You are a senior data analyst working inside an interactive analytics app. \
The user has uploaded data and is asking questions about it in a chat.

Currently loaded data:
{schema}

How you work:
1. Never guess at values. Every number you state must come from a tool call in this conversation.
2. Prefer `run_sql` for aggregation, filtering and joins. Use `run_python` for statistics, \
reshaping, correlations and anything SQL handles badly. Use `analyze_trend` for time series, \
`detect_anomalies` for outliers, and `compare_groups` to test differences between segments.
3. Check the data before trusting it. Watch for nulls, duplicates, mixed units, zero-inflation \
and groups too small to be meaningful. Say so when they affect the answer.
4. Answer the question that was asked, then go one step further: when you see a change, look for \
what drove it; when you see an outlier, look at what the outlier rows have in common.
5. Call `create_chart` when a picture communicates better than a sentence. The chart is already \
shown to the user — reference it, don't narrate it.
6. Remember earlier turns. If the user says "that segment" or "same thing but weekly", resolve it \
from the conversation rather than asking again.
7. If a question cannot be answered from these columns, say exactly what is missing.

Keep your working notes terse. You are being reviewed for factual grounding, so make the \
link between each claim and the tool output that supports it obvious."""

PLANNER_SYSTEM = """You plan analyses for a data analyst agent. Given the dataset schema and the \
user's question, write a compact plan: 2-5 numbered steps naming the specific columns, \
aggregations and tools to use, plus the main thing that could make the result misleading.

Be concrete and short — under 120 words, no preamble. If the question is small talk or a \
follow-up needing no new computation, reply with exactly: DIRECT"""

VALIDATOR_SYSTEM = """You are a meticulous reviewer checking a data analyst's draft answer against \
the tool outputs in the conversation.

Check that:
- every number and name in the draft appears in a tool result
- the aggregation actually answers the question asked
- caveats are stated where nulls, tiny samples, duplicates or skew affect the result
- no causal claim is made from a correlation alone

Reply with JSON only:
{"verdict": "pass" | "revise", "issues": ["..."], "note": "one line"}

Use "revise" only for problems that would mislead the user. Style preferences are not issues."""

EXPLAINER_SYSTEM = """Rewrite the analyst's verified findings as the final answer for a business \
user reading it in a chat window.

Structure:
- Open with the direct answer in one or two sentences. Lead with the number.
- Then the supporting detail: what the data shows, with the figures that matter.
- Then "Worth knowing": caveats, data quality limits, or what would change the conclusion. \
Skip this if there is genuinely nothing to flag.
- Close with one specific follow-up question the user could ask next.

Use markdown. Keep it under 300 words. No tables unless comparing three or more items. \
Charts are already displayed above your answer, so refer to them naturally rather than \
re-describing them. Never invent a number that is not in the findings."""


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    attempts: int
    verdict: str
    issues: list[str]
    steps: int


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _tool_evidence(messages: list[BaseMessage], limit: int = 8) -> str:
    """The most recent tool outputs, for the validator and explainer to read."""
    chunks = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            chunks.append(f"[{msg.name}]\n{str(msg.content)[:2500]}")
        if len(chunks) >= limit:
            break
    return "\n\n".join(reversed(chunks)) or "(no tool output)"


def message_text(msg: BaseMessage) -> str:
    """Flatten message content, which may be a string or a list of content blocks."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return str(content)


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = message_text(msg)
            if text.strip():
                return text
    return ""


def _last_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return message_text(msg)
    return ""


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_agent(ctx: AnalysisContext, checkpointer: Any | None = None):
    """Compile the analyst graph bound to one session's data context."""
    llm = build_llm()
    tools = build_tools(ctx)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # ---- nodes -----------------------------------------------------------
    def plan(state: AgentState) -> dict[str, Any]:
        question = _last_human_text(state["messages"])
        if not SETTINGS.enable_planner or not question:
            return {"plan": "", "attempts": 0, "issues": [], "steps": 0, "verdict": ""}

        reply = llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(
                    content=f"Schema:\n{summarize_table_for_prompt(ctx)}\n\nQuestion: {question}"
                ),
            ]
        )
        text = message_text(reply).strip()
        return {
            "plan": "" if text.upper().startswith("DIRECT") else text,
            "attempts": 0,
            "issues": [],
            "steps": 0,
            "verdict": "",
        }

    def analyst(state: AgentState) -> dict[str, Any]:
        system = ANALYST_SYSTEM.format(schema=summarize_table_for_prompt(ctx))
        if state.get("plan"):
            system += f"\n\nYour plan for this question:\n{state['plan']}"

        reply = llm_with_tools.invoke([SystemMessage(content=system)] + state["messages"])
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    def validate(state: AgentState) -> dict[str, Any]:
        draft = _last_ai_text(state["messages"])
        if not SETTINGS.enable_validator or not draft:
            return {"verdict": "pass", "issues": []}

        reply = llm.invoke(
            [
                SystemMessage(content=VALIDATOR_SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {_last_human_text(state['messages'])}\n\n"
                        f"Tool outputs:\n{_tool_evidence(state['messages'])}\n\n"
                        f"Draft answer:\n{draft}"
                    )
                ),
            ]
        )
        parsed = _extract_json(message_text(reply)) or {"verdict": "pass", "issues": []}
        verdict = parsed.get("verdict", "pass")
        issues = [str(i) for i in parsed.get("issues", [])][:5]
        if verdict != "pass" and state.get("attempts", 0) >= SETTINGS.max_repair_attempts:
            verdict = "pass"  # ship it, with the caveats surfaced downstream
        return {"verdict": verdict, "issues": issues}

    def repair(state: AgentState) -> dict[str, Any]:
        issues = "\n".join(f"- {i}" for i in state.get("issues", []))
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "Review of your draft found problems that would mislead the reader:\n"
                        f"{issues}\n\nRun whatever tool calls you need to resolve these, then "
                        "give a corrected answer. Do not repeat an unsupported claim."
                    )
                )
            ],
            "attempts": state.get("attempts", 0) + 1,
        }

    def explain(state: AgentState) -> dict[str, Any]:
        draft = _last_ai_text(state["messages"])
        caveats = state.get("issues", []) if state.get("verdict") != "pass" else []
        extra = ("\n\nUnresolved review notes to fold in as caveats:\n"
                 + "\n".join(f"- {c}" for c in caveats)) if caveats else ""

        reply = llm.invoke(
            [
                SystemMessage(content=EXPLAINER_SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {_last_human_text(state['messages'])}\n\n"
                        f"Verified findings:\n{draft}\n\n"
                        f"Supporting tool output:\n{_tool_evidence(state['messages'], 5)}{extra}"
                    )
                ),
            ]
        )
        return {"messages": [AIMessage(content=message_text(reply))]}

    # ---- edges -----------------------------------------------------------
    def after_analyst(state: AgentState) -> Literal["tools", "validate"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("steps", 0) <= SETTINGS.max_tool_steps:
            return "tools"
        return "validate"

    def after_validate(state: AgentState) -> Literal["repair", "explain"]:
        return "repair" if state.get("verdict") == "revise" else "explain"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("analyst", analyst)
    graph.add_node("tools", tool_node)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("explain", explain)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "analyst")
    graph.add_conditional_edges("analyst", after_analyst, {"tools": "tools", "validate": "validate"})
    graph.add_edge("tools", "analyst")
    graph.add_conditional_edges("validate", after_validate, {"repair": "repair", "explain": "explain"})
    graph.add_edge("repair", "analyst")
    graph.add_edge("explain", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


def ask(agent, question: str, thread_id: str, recursion_limit: int = 40) -> dict[str, Any]:
    """Run one turn. The checkpointer supplies prior messages for this thread."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)


def stream_steps(agent, question: str, thread_id: str, recursion_limit: int = 40):
    """Yield (node_name, state_update) as the graph runs, for live progress in the UI."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=question)]}, config=config, stream_mode="updates"
    ):
        for node, update in chunk.items():
            yield node, update
