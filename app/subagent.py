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
