import asyncio
import os
import logging
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from app.logger import get_logger
from app.registry import AgentRegistry
from app.divider import divide_task
from app.subagent import spawn_subagent, run_subagent, destroy_subagent

logger = get_logger("mitosys.orchestrator")

MODEL = "gpt-4o"

_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a senior analyst. "
    "You will be given several research results from specialist agents. "
    "Synthesize them into one cohesive, well-structured final answer. "
    "Remove redundancy, preserve key insights, and write clearly."
)


class _LogCapture(logging.Handler):
    """Captures log records from the mitosys logger into a list."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


async def run_mitosys(task: str) -> dict:
    # Attach a capture handler for this run so we can return logs in the response
    capture = _LogCapture()
    capture.setFormatter(logging.Formatter("%(asctime)s | MITOSYS | %(message)s", "%Y-%m-%d %H:%M:%S"))
    root_logger = logging.getLogger("mitosys")
    root_logger.addHandler(capture)

    try:
        api_key = os.environ["OPENAI_API_KEY"]

        # ── Stage 1: RECEIVE ────────────────────────────────────────────────
        logger.info(f"Parent agent received the task: '{task}'")

        # ── Stage 2: DIVIDE ─────────────────────────────────────────────────
        logger.info("Parent agent is consulting the LLM to divide the task into sub-tasks.")
        planner_client = OpenAIChatCompletionClient(model=MODEL, api_key=api_key)
        sub_tasks = await divide_task(task, planner_client)
        await planner_client.close()
        logger.info(f"The planner client has been closed after task division.")

        # ── Stage 3: SPAWN ──────────────────────────────────────────────────
        registry = AgentRegistry()
        logger.info(f"Parent agent is spawning {len(sub_tasks)} sub-agent(s).")

        # Each sub-agent gets its own model client so it can be closed independently
        spawned: list[tuple[AssistantAgent, object, str]] = []
        for i, subtask in enumerate(sub_tasks, 1):
            name = f"subagent_{i}"
            agent, client = await spawn_subagent(name, subtask, MODEL, api_key)
            registry.add(name, agent)
            spawned.append((agent, client, subtask))

        # ── Stage 4: EXECUTE (concurrent) ───────────────────────────────────
        logger.info(
            f"All {len(spawned)} sub-agent(s) are now working concurrently on their sub-tasks."
        )


        async def _execute(agent, client, subtask):
            result_text = await run_subagent(agent, subtask)
            return {"agent": agent.name, "name": agent.name, "subtask": subtask, "result": result_text, "client": client}

        # async def _execute(agent, client, subtask):
        #     result_text = await run_subagent(agent, subtask)
        #     return {"agent": agent.name, "subtask": subtask, "result": result_text, "client": client}

        raw_results = await asyncio.gather(
            *[_execute(agent, client, subtask) for agent, client, subtask in spawned]
        )

        # ── Stage 5: COLLECT ─────────────────────────────────────────────────
        logger.info("Parent agent is collecting all sub-agent results.")
        sub_results = [
            {"agent": r["agent"], "subtask": r["subtask"], "result": r["result"]}
            for r in raw_results
        ]
        for r in sub_results:
            logger.info(f"Collected result from '{r['agent']}': work complete.")

        # ── Stage 6: DESTROY ─────────────────────────────────────────────────
        logger.info("Parent agent is beginning the destruction of all sub-agents.")
        for r in raw_results:
            await destroy_subagent(r["name"], r["client"], registry)

        # ── Synthesize ───────────────────────────────────────────────────────
        logger.info("Parent agent is synthesizing all results into a final answer.")
        combined = "\n\n".join(
            f"[{r['agent']} on '{r['subtask']}']\n{r['result']}" for r in sub_results
        )
        synthesis_prompt = (
            f"Original task: {task}\n\n"
            f"Sub-agent results:\n{combined}\n\n"
            "Please synthesize the above into one clear, final answer."
        )

        synth_client = OpenAIChatCompletionClient(model=MODEL, api_key=api_key)
        synthesizer = AssistantAgent(
            name="synthesizer",
            model_client=synth_client,
            system_message=_SYNTHESIZER_SYSTEM_PROMPT,
        )
        synth_result = await synthesizer.run(task=synthesis_prompt)
        raw_synth = synth_result.messages[-1].content
        if isinstance(raw_synth, list):
            final_answer = " ".join(
                p.text if hasattr(p, "text") else str(p) for p in raw_synth
            )
        else:
            final_answer = str(raw_synth)

        await synth_client.close()
        logger.info("The synthesizer agent has completed its work and has been destroyed.")

        # ── Stage 7: RETURN ──────────────────────────────────────────────────
        logger.info("Mitosys lifecycle complete. Returning the final answer to the caller.")

        return {
            "task": task,
            "sub_tasks": sub_tasks,
            "sub_results": sub_results,
            "final_answer": final_answer,
            "log": list(capture.lines),
        }

    finally:
        root_logger.removeHandler(capture)
