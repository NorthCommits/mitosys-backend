import json

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from app.logger import get_logger
from app.registry import AgentRegistry

logger = get_logger("mitosys.subagent")

_SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused research and writing specialist. "
    "Complete the given sub-task thoroughly and concisely. "
    "Provide a clear, well-structured response. "
    "Do not ask clarifying questions — do your best with what you have."
)

_SELF_ASSESS_PROMPT = (
    "You are a sub-agent inside Mitosys, a controlled multi-agent system inspired by "
    "biological mitosis. Uncontrolled division is cancer — you must LEAN STRONGLY toward "
    "executing your sub-task directly.\n\n"
    "You are at depth {depth} of the recursion tree. "
    "Your sub-task has an effort hint of {effort_hint}/10.\n\n"
    "Assess honestly: can you complete this sub-task alone to a high standard, "
    "or is it genuinely too broad for a single focused agent?\n\n"
    "IMPORTANT BIAS: Execute directly unless the sub-task CLEARLY spans several "
    "distinct, independent domains each requiring deep, separate treatment. "
    "A complex but unified topic does NOT warrant division.\n\n"
    "Output STRICT JSON — exactly one of:\n"
    "  {{\"decision\": \"execute\"}}\n"
    "  {{\"decision\": \"propose_division\", \"effort\": <int 1-10>, "
    "\"proposed_children\": <int 2-5>, \"reasoning\": \"<one sentence>\"}}\n\n"
    "Output ONLY the JSON. No preamble, no markdown fences."
)


def _extract_text(result) -> str:
    content = result.messages[-1].content
    if isinstance(content, list):
        return " ".join(p.text if hasattr(p, "text") else str(p) for p in content)
    return str(content)


async def spawn_subagent(name: str, subtask: str, model: str, api_key: str) -> tuple:
    model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
    agent = AssistantAgent(
        name=name,
        model_client=model_client,
        system_message=_SUBAGENT_SYSTEM_PROMPT,
    )
    logger.info(f"Sub-agent '{name}' has been born and assigned its sub-task: {subtask}")
    return agent, model_client


async def run_subagent(agent: AssistantAgent, subtask: str) -> str:
    logger.info(f"Sub-agent '{agent.name}' is starting work on its sub-task.")
    result = await agent.run(task=subtask)
    raw_content = result.messages[-1].content
    if isinstance(raw_content, list):
        answer = " ".join(
            part.text if hasattr(part, "text") else str(part) for part in raw_content
        )
    else:
        answer = str(raw_content)
    logger.info(f"Sub-agent '{agent.name}' has finished its sub-task and returned its result.")
    return answer


async def self_assess(
    agent: AssistantAgent,
    subtask: str,
    effort_hint: int,
    depth: int,
    model_client,
) -> dict:
    """
    One LLM call where the sub-agent decides: execute directly or propose further division.
    Uses the same model_client as the sub-agent (no extra client to manage).
    """
    system_msg = _SELF_ASSESS_PROMPT.format(depth=depth, effort_hint=effort_hint)
    assessor = AssistantAgent(
        name=f"{agent.name}_assessor",
        model_client=model_client,
        system_message=system_msg,
    )
    logger.info(
        f"Sub-agent '{agent.name}' is self-assessing its sub-task (effort hint {effort_hint}/10)."
    )
    result = await assessor.run(
        task=f"Assess whether to execute directly or propose division for this sub-task:\n{subtask}"
    )
    raw = _extract_text(result).strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
        raw = "\n".join(inner).strip()

    try:
        parsed = json.loads(raw)
        decision = parsed.get("decision", "execute")
        if decision not in ("execute", "propose_division"):
            raise ValueError(f"Unknown decision: '{decision}'")
    except Exception as exc:
        logger.warning(
            f"Sub-agent '{agent.name}' self-assessment parse failure ({exc}). "
            "Defaulting to execute."
        )
        return {"decision": "execute"}

    if decision == "execute":
        logger.info(
            f"Sub-agent '{agent.name}' will execute directly without further division."
        )
    else:
        logger.info(
            f"Sub-agent '{agent.name}' proposes further division: "
            f"effort {parsed.get('effort', '?')}/10, "
            f"requesting {parsed.get('proposed_children', '?')} children. "
            f"Reasoning: {parsed.get('reasoning', '')}"
        )

    return parsed


async def destroy_subagent(
    name: str,
    model_client: OpenAIChatCompletionClient,
    registry: AgentRegistry,
) -> None:
    await model_client.close()
    registry.remove(name)
    logger.info(
        f"Sub-agent '{name}' has been destroyed. "
        f"{registry.active_count()} sub-agent(s) remaining."
    )
