import re

from autogen_agentchat.agents import AssistantAgent
from app.logger import get_logger

logger = get_logger("mitosys.divider")


def _build_prompt(target_count: int) -> str:
    return (
        "You are a task decomposition specialist inside Mitosys, a controlled multi-agent "
        "system inspired by biological mitosis. You are the cell division mechanism: "
        "divide precisely and cleanly, never over-dividing.\n\n"
        f"CRITICAL: Produce EXACTLY {target_count} sub-task(s). No more, no fewer.\n\n"
        "RULES:\n"
        "1. Every sub-task must be fully self-contained and executable in isolation.\n"
        "2. Sub-tasks must NOT depend on each other's output.\n"
        "3. Each sub-task must carry enough context to be completed standalone — "
        "write it as if assigning to a person who cannot see any other sub-task.\n"
        "4. Bad sub-tasks: 'Proofread the report', 'Organize the findings', 'Summarize the above'.\n\n"
        "OUTPUT FORMAT — one line per sub-task, no numbering, no bullets:\n"
        "<complete self-contained sub-task sentence> | <effort_hint 1-10>\n\n"
        "Where effort_hint reflects the cognitive complexity of THAT sub-task alone "
        "(1=trivial, 10=very hard).\n\n"
        f"Example output for target_count=3 on 'Write a report on green tea':\n"
        "Research and write about the antioxidant properties of green tea and their health impact. | 4\n"
        "Research and write about green tea's effects on heart health and cholesterol. | 4\n"
        "Research and write about green tea's role in weight management and metabolism. | 4\n\n"
        f"Produce exactly {target_count} line(s). No preamble, no explanation, just the lines."
    )


async def divide_task(task: str, model_client, target_count: int = 3) -> list[dict]:
    planner = AssistantAgent(
        name="task_planner",
        model_client=model_client,
        system_message=_build_prompt(target_count),
    )

    result = await planner.run(
        task=f"Decompose this task into exactly {target_count} sub-task(s): {task}"
    )

    raw_content = result.messages[-1].content
    if isinstance(raw_content, list):
        raw_text = " ".join(
            part.text if hasattr(part, "text") else str(part) for part in raw_content
        )
    else:
        raw_text = str(raw_content)

    sub_tasks: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading list markers if the LLM disobeyed formatting instructions
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        line = re.sub(r"^\s*[-*•]\s*", "", line)

        if "|" in line:
            parts = line.rsplit("|", 1)
            subtask_text = parts[0].strip()
            try:
                effort_hint = max(1, min(10, int(parts[1].strip())))
            except (ValueError, IndexError):
                effort_hint = 5
        else:
            subtask_text = line
            effort_hint = 5

        if subtask_text:
            sub_tasks.append({"subtask": subtask_text, "effort_hint": effort_hint})

    if not sub_tasks:
        logger.warning(
            "Divider returned no parseable sub-tasks. Falling back to single sub-task."
        )
        sub_tasks = [{"subtask": task, "effort_hint": 5}]

    logger.info(f"Parent divided the task into {len(sub_tasks)} sub-task(s).")
    for i, st in enumerate(sub_tasks, 1):
        logger.info(f"  Sub-task {i} (effort {st['effort_hint']}/10): {st['subtask']}")

    return sub_tasks
