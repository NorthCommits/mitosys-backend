from autogen_agentchat.agents import AssistantAgent
from app.logger import get_logger

logger = get_logger("mitosys.divider")

_DIVIDER_SYSTEM_PROMPT = (
    "You are a task decomposition specialist. "
    "When given a task, break it into clear, concrete, independent sub-tasks. "
    "CRITICAL RULE: Every sub-task must be fully self-contained and executable in isolation. "
    "Each sub-task must NOT depend on the output or results of any other sub-task. "
    "Each sub-task must carry enough context within its own sentence to be completed standalone. "
    "Think of each sub-task as being assigned to a different person who cannot see what others are doing. "
    "Good example for 'Write a report on green tea': "
    "'Research and write about the antioxidant properties of green tea and their impact on health.' "
    "'Research and write about green tea's effects on heart health and cholesterol levels.' "
    "'Research and write about green tea's role in weight management and metabolism.' "
    "Bad example (DO NOT do this): 'Proofread the report.' or 'Organize the findings.' "
    "Reply with exactly one sub-task per line. "
    "Each line must be a complete, actionable sentence with full context. "
    "Output between 3 and 5 sub-tasks. "
    "Do not use numbering, bullets, or any symbols — plain sentences only. "
    "Do not include any explanation or preamble, just the sub-task sentences."
)


async def divide_task(task: str, model_client) -> list[str]:
    planner = AssistantAgent(
        name="task_planner",
        model_client=model_client,
        system_message=_DIVIDER_SYSTEM_PROMPT,
    )

    result = await planner.run(task=f"Decompose this task into sub-tasks: {task}")

    raw_content = result.messages[-1].content
    if isinstance(raw_content, list):
        raw_text = " ".join(
            part.text if hasattr(part, "text") else str(part) for part in raw_content
        )
    else:
        raw_text = str(raw_content)

    sub_tasks = [line.strip() for line in raw_text.splitlines() if line.strip()]

    logger.info(f"Parent divided the task into {len(sub_tasks)} sub-tasks.")
    for i, st in enumerate(sub_tasks, 1):
        logger.info(f"  Sub-task {i}: {st}")

    return sub_tasks
