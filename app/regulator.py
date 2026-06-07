"""
Parent regulator — the p53 of Mitosys.
Approves or denies sub-agent division proposals.
Uncontrolled division is cancer. This module is the immune checkpoint.
"""
import json

from autogen_agentchat.agents import AssistantAgent
from app.logger import get_logger

logger = get_logger("mitosys.regulator")

_REGULATOR_SYSTEM_PROMPT = (
    "You are the Parent Regulator inside Mitosys, a controlled multi-agent system "
    "inspired by biological mitosis. Your role is analogous to the p53 protein in biology: "
    "you grant or deny permission for a cell (sub-agent) to divide further. "
    "Without you, division is uncontrolled and the system becomes cancerous.\n\n"
    "A sub-agent has proposed dividing its sub-task into N children. You receive:\n"
    "  - The original task (the root purpose)\n"
    "  - All current sibling sub-tasks (what other agents are already handling)\n"
    "  - The current recursion depth\n"
    "  - The current total agent count spawned so far\n"
    "  - The proposal: the sub-task text, the sub-agent's own effort score, "
    "proposed number of children, and the sub-agent's reasoning\n\n"
    "APPROVE only if ALL of the following hold:\n"
    "  - The sub-task is genuinely too complex for a single focused agent\n"
    "  - The proposed children would be truly independent and parallel-safe\n"
    "  - The recursion depth is still reasonable (deeper = more dangerous)\n"
    "  - Total agent count remains proportionate to the root task complexity\n\n"
    "DENY when:\n"
    "  - The sub-task is well-scoped and one focused agent can handle it\n"
    "  - The sub-agent's reasoning is weak, evasive, or circular\n"
    "  - Further division would fragment the result rather than improve it\n"
    "  - The tree is already deep and the marginal gain is small\n\n"
    "BIAS STRONGLY TOWARD DENIAL. Only approve when the case is genuinely compelling "
    "and the sub-task spans multiple truly independent sub-domains.\n\n"
    "Output STRICT JSON with exactly two fields:\n"
    "  {\"approved\": true|false, \"reasoning\": \"...\"}\n\n"
    "Output ONLY the JSON. No preamble, no markdown fences."
)


def _extract_text(result) -> str:
    content = result.messages[-1].content
    if isinstance(content, list):
        return " ".join(p.text if hasattr(p, "text") else str(p) for p in content)
    return str(content)


async def evaluate_proposal(
    original_task: str,
    all_current_subtasks: list[str],
    current_depth: int,
    current_total_agents: int,
    proposal: dict,
    model_client,
) -> dict:
    siblings = "\n".join(f"  - {s}" for s in all_current_subtasks)
    prompt = (
        f"Original task: {original_task}\n\n"
        f"Current sibling sub-tasks (already being handled in parallel):\n{siblings}\n\n"
        f"Current recursion depth: {current_depth}\n"
        f"Current total agents spawned so far: {current_total_agents}\n\n"
        f"Division proposal from sub-agent '{proposal['agent']}':\n"
        f"  Sub-task: {proposal['subtask']}\n"
        f"  Self-assessed effort: {proposal['effort']}/10\n"
        f"  Proposed number of children: {proposal['proposed_children']}\n"
        f"  Sub-agent's reasoning: {proposal['reasoning']}\n\n"
        "Should this division be approved? Remember: bias toward denial."
    )

    regulator = AssistantAgent(
        name="regulator",
        model_client=model_client,
        system_message=_REGULATOR_SYSTEM_PROMPT,
    )
    result = await regulator.run(task=prompt)
    raw = _extract_text(result).strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        raw = "\n".join(inner).strip()

    try:
        parsed = json.loads(raw)
        approved = bool(parsed["approved"])
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as exc:
        logger.warning(
            f"Regulator parse failure ({exc}). Defaulting to denial for safety."
        )
        return {"approved": False, "reasoning": "parse_failure_fallback_denial"}

    if approved:
        logger.info(
            f"Parent regulator approved '{proposal['agent']}'s division proposal. "
            f"Reasoning: {reasoning}"
        )
    else:
        logger.info(
            f"Parent regulator denied '{proposal['agent']}'s division proposal "
            f"because: {reasoning}"
        )

    return {"approved": approved, "reasoning": reasoning}
