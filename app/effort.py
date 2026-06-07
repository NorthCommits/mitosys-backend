"""
Effort evaluator — the growth-factor signal of Mitosys.
Decides how many sub-agents a task warrants before any division begins.
Uncontrolled division is cancer. This module prevents it at the source.
"""
import json

from autogen_agentchat.agents import AssistantAgent
from app.logger import get_logger

logger = get_logger("mitosys.effort")


_EFFORT_SYSTEM_PROMPT = (
    "You are the Effort Evaluator inside Mitosys, a controlled multi-agent system "
    "inspired by biological mitosis. Your role is analogous to a growth-factor signal "
    "in a cell: you decide how much division is warranted — no more and no less.\n\n"
    "UNCONTROLLED DIVISION IS CANCER. Recommending too many agents wastes resources "
    "and produces fragmented, low-quality output. Recommending too few leaves the task "
    "under-served. When in doubt, prefer fewer agents.\n\n"
    "You must reason along TWO dimensions before deciding:\n"
    "  BREADTH — how many genuinely independent angles, sub-domains, or parallel "
    "threads does the task contain?\n"
    "  DEPTH   — how much cognitive effort does each angle demand on its own? "
    "Is it a one-line answer, a paragraph, a section, or a full investigation?\n\n"
    "A task with 2 angles where each demands deep, multi-part investigation may "
    "warrant 4 agents (two per angle). A task with 6 shallow angles may warrant "
    "only 2 agents (one agent can comfortably handle three shallow points). "
    "Breadth alone does not determine agent count — breadth × depth does.\n\n"
    "For the given task, return STRICT JSON with exactly these fields:\n"
    "  score              integer 1-10, the OVERALL cognitive load (breadth combined with depth)\n"
    "  breadth            integer 1-10, how many independent angles the task spans\n"
    "  depth              integer 1-10, how heavy each angle is on average\n"
    "  recommended_agents integer >= 1, derived from breadth and depth together\n"
    "  reasoning          one or two plain-English sentences explaining the trade-off you made\n\n"
    "Guidelines (use as anchors, not as a lookup table):\n"
    "  - Trivial task (haiku, single fact, one-line answer): score 1-2, agents 1\n"
    "  - Light task with one clear thread: score 3-4, agents 1-2\n"
    "  - Moderate task with several independent angles of moderate depth: score 5-7, agents 3-5\n"
    "  - Heavy task spanning many independent and deep sub-domains: score 8-10, agents 5-8\n"
    "  - Above 8 agents only when the task is genuinely encyclopaedic AND each angle is independently substantial\n"
    "  - If angles depend on each other sequentially, parallelism gives no benefit — prefer 1 agent\n\n"
    "Example output:\n"
    "  {\"score\": 7, \"breadth\": 4, \"depth\": 6, \"recommended_agents\": 4, "
    "\"reasoning\": \"The task has four independent angles, each requiring a substantive "
    "paragraph of analysis. One agent per angle gives each the depth it deserves without fragmenting.\"}\n\n"
    "Output ONLY the JSON object. No preamble, no markdown fences."
)

# _EFFORT_SYSTEM_PROMPT = (
#     "You are the Effort Evaluator inside Mitosys, a controlled multi-agent system "
#     "inspired by biological mitosis. Your role is analogous to a growth-factor signal "
#     "in a cell: you decide how much division is warranted — no more and no less.\n\n"
#     "UNCONTROLLED DIVISION IS CANCER. Recommending too many agents wastes resources "
#     "and produces fragmented, low-quality output. Recommending too few leaves the task "
#     "under-served. Your goal is the MINIMUM agent count that genuinely improves the result.\n\n"
#     "For the given task, return STRICT JSON with exactly three fields:\n"
#     "  score              integer 1-10, the cognitive load of the task\n"
#     "  recommended_agents integer >= 1, how many parallel sub-agents would materially "
#     "improve the output compared to one agent doing it alone\n"
#     "  reasoning          a brief plain-English explanation (one sentence)\n\n"
#     "Heuristics:\n"
#     "  - Single-fact or single-paragraph tasks: 1 agent\n"
#     "  - Tasks with 2-3 clear independent angles: 2-3 agents\n"
#     "  - Tasks spanning many distinct independent sub-domains: 4-6 agents\n"
#     "  - Rarely above 6 unless the task is genuinely encyclopaedic\n"
#     "  - If sub-tasks would depend on each other sequentially, prefer 1 agent "
#     "(parallelism gives no benefit)\n\n"
#     "Output ONLY the JSON object. No preamble, no markdown fences."
# )


def _extract_text(result) -> str:
    content = result.messages[-1].content
    if isinstance(content, list):
        return " ".join(p.text if hasattr(p, "text") else str(p) for p in content)
    return str(content)


async def evaluate_effort(task: str, model_client) -> dict:
    evaluator = AssistantAgent(
        name="effort_evaluator",
        model_client=model_client,
        system_message=_EFFORT_SYSTEM_PROMPT,
    )
    result = await evaluator.run(task=f"Evaluate the effort required for this task: {task}")
    raw = _extract_text(result).strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        raw = "\n".join(inner).strip()

    try:
        parsed = json.loads(raw)
        score = max(1, min(10, int(parsed["score"])))
        recommended_agents = max(1, int(parsed["recommended_agents"]))
        reasoning = str(parsed.get("reasoning", ""))
        breadth = max(1, min(10, int(parsed["breadth"]))) if "breadth" in parsed else None
        depth_val = max(1, min(10, int(parsed["depth"]))) if "depth" in parsed else None
    except Exception as exc:
        logger.warning(
            f"Effort evaluator parse failure ({exc}). Falling back to score=5, agents=3."
        )
        return {"score": 5, "recommended_agents": 3, "reasoning": "parse_failure_fallback"}

    logger.info(
        f"Effort evaluator scored the task {score}/10 and recommends "
        f"{recommended_agents} sub-agent(s)."
    )
    result = {"score": score, "recommended_agents": recommended_agents, "reasoning": reasoning}
    if breadth is not None:
        result["breadth"] = breadth
    if depth_val is not None:
        result["depth"] = depth_val
    return result
