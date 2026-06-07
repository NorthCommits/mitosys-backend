"""
Mitosys Orchestrator — the controlled cell division engine.

"Uncontrolled division is cancer. Mitosys is controlled mitosis."

Every division decision flows through the Effort Evaluator (growth-factor signal)
and the Parent Regulator (p53). The sanity ceilings below are parse-failure backstops
only — they should almost never trigger under normal LLM behavior. The prompts are
the primary safety mechanism.
"""
import asyncio
import logging
import os
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from app.divider import divide_task
from app.effort import evaluate_effort
from app.logger import get_logger
from app.regulator import evaluate_proposal
from app.registry import AgentRegistry
from app.subagent import destroy_subagent, run_subagent, self_assess, spawn_subagent

logger = get_logger("mitosys.orchestrator")

MODEL = "gpt-4o"

# Sanity backstops — triggered ONLY on malformed LLM output (e.g. hallucinated depth=999).
# These are NOT creative limits. Set wide so normal behavior never touches them.
MAX_DEPTH_FALLBACK = 5
MAX_TOTAL_AGENTS_FALLBACK = 100

_SYNTHESIZER_SYSTEM_PROMPT = (
    "You are a senior analyst. "
    "You will be given several research results from specialist agents. "
    "Synthesize them into one cohesive, well-structured final answer. "
    "Remove redundancy, preserve key insights, and write clearly."
)


@dataclass
class GlobalState:
    total_agents_spawned: int = 0
    max_depth_reached: int = 0


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class _Sentinel:
    """Signals completion of the background lifecycle task to the streaming generator."""
    def __init__(self, result=None, error: str = None):
        self.result = result
        self.error = error


async def _synthesize(task: str, sub_results: list[dict], api_key: str) -> str:
    if len(sub_results) == 1:
        return sub_results[0]["result"]
    combined = "\n\n".join(
        f"[{r['agent']} on '{r['subtask']}']\n{r['result']}" for r in sub_results
    )
    prompt = (
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
    result = await synthesizer.run(task=prompt)
    content = result.messages[-1].content
    if isinstance(content, list):
        answer = " ".join(p.text if hasattr(p, "text") else str(p) for p in content)
    else:
        answer = str(content)
    await synth_client.close()
    logger.info("Synthesizer agent completed its work and has been destroyed.")
    return answer


async def _run_lifecycle_core(
    task: str,
    depth: int,
    global_state: GlobalState,
    api_key: str,
    name_prefix: str,
    on_event,  # async callable(event: dict) | None
) -> dict:
    """
    Recursive lifecycle worker. Shared by both the batch and streaming paths.

    on_event is called for every lifecycle event. In the streaming path it pushes
    to a queue; in the batch path it is None (no-op). All logger.info calls go to
    stdout and to _LogCapture regardless of on_event.
    """

    async def emit(event: dict):
        if on_event:
            await on_event(event)

    async def log_emit(msg: str):
        logger.info(msg)
        await emit({"type": "log", "message": msg})

    if depth > global_state.max_depth_reached:
        global_state.max_depth_reached = depth

    # ── EFFORT EVALUATION ──────────────────────────────────────────────────
    await log_emit(f"Effort evaluator is assessing the task at depth {depth}.")

    effort_client = OpenAIChatCompletionClient(model=MODEL, api_key=api_key)
    effort = await evaluate_effort(task, effort_client)
    await effort_client.close()

    await log_emit(
        f"Effort evaluator scored the task {effort['score']}/10 and recommends "
        f"{effort['recommended_agents']} sub-agent(s). Reasoning: {effort['reasoning']}"
    )
    await emit({"type": "effort", "recursion_depth": depth, **effort})

    n_agents = effort["recommended_agents"]

    # ── DIVIDE ─────────────────────────────────────────────────────────────
    await log_emit(
        f"Parent agent is consulting the LLM to divide the task into "
        f"{n_agents} sub-task(s) at depth {depth}."
    )

    planner_client = OpenAIChatCompletionClient(model=MODEL, api_key=api_key)
    subtask_dicts = await divide_task(task, planner_client, target_count=n_agents)
    await planner_client.close()

    await emit({"type": "subtasks", "subtasks": subtask_dicts, "depth": depth})
    await log_emit(
        f"Parent divided the task into {len(subtask_dicts)} sub-task(s) at depth {depth}."
    )

    all_subtask_texts = [d["subtask"] for d in subtask_dicts]

    # ── SPAWN ──────────────────────────────────────────────────────────────
    registry = AgentRegistry()
    await log_emit(
        f"Parent agent is spawning {len(subtask_dicts)} sub-agent(s) at depth {depth + 1}."
    )

    spawned: list[tuple] = []
    for i, sd in enumerate(subtask_dicts, 1):
        name = f"{name_prefix}_{i}" if name_prefix else f"subagent_{i}"

        if global_state.total_agents_spawned >= MAX_TOTAL_AGENTS_FALLBACK:
            msg = (
                f"Cancer check triggered: total agent count would reach "
                f"{MAX_TOTAL_AGENTS_FALLBACK}. Not spawning '{name}'. Division refused."
            )
            logger.warning(msg)
            await emit({"type": "cancer_check", "agent": name, "reason": "total_agents_ceiling_at_spawn"})
            await emit({"type": "log", "message": msg})
            break

        agent, client = await spawn_subagent(name, sd["subtask"], MODEL, api_key)
        registry.add(name, agent)
        global_state.total_agents_spawned += 1
        spawned.append((agent, client, sd["subtask"], sd["effort_hint"]))

        await emit({
            "type": "spawn",
            "agent": name,
            "parent": name_prefix or "root",
            "depth": depth + 1,
            "subtask": sd["subtask"],
        })
        await emit({"type": "log", "message": f"Sub-agent '{name}' has been born and assigned its sub-task."})

    # ── EXECUTE (concurrent — all agents at this depth work in parallel) ───
    await log_emit(
        f"All {len(spawned)} sub-agent(s) at depth {depth + 1} are now working concurrently."
    )

    async def _work_one(agent, client, subtask: str, effort_hint: int) -> dict:
        name = agent.name
        decision_label = "execute"
        child_nodes: list = []

        try:
            # Self-assessment: should this agent execute or propose further division?
            assessment = await self_assess(agent, subtask, effort_hint, depth + 1, client)
            await emit({"type": "self_assessment", "agent": name, **assessment})

            if assessment.get("decision") == "propose_division":
                proposal = {
                    "agent": name,
                    "subtask": subtask,
                    "effort": assessment.get("effort", effort_hint),
                    "proposed_children": assessment.get("proposed_children", 2),
                    "reasoning": assessment.get("reasoning", ""),
                }
                await emit({"type": "proposal", **proposal})

                # Sanity ceiling — backstop only, not a creative limit
                proposed_n = proposal["proposed_children"]
                would_exceed_depth = (depth + 1) >= MAX_DEPTH_FALLBACK
                would_exceed_agents = (
                    global_state.total_agents_spawned + proposed_n > MAX_TOTAL_AGENTS_FALLBACK
                )

                if would_exceed_depth or would_exceed_agents:
                    reason = (
                        f"depth {depth + 1} would reach sanity ceiling of {MAX_DEPTH_FALLBACK}"
                        if would_exceed_depth
                        else f"total agents would exceed {MAX_TOTAL_AGENTS_FALLBACK}"
                    )
                    msg = (
                        f"Cancer check triggered for '{name}': {reason}. "
                        "Forcing direct execution."
                    )
                    logger.warning(msg)
                    await emit({"type": "cancer_check", "agent": name, "reason": reason})
                    await emit({"type": "log", "message": msg})
                    result_text = await run_subagent(agent, subtask)
                    decision_label = "execute_after_cancer_check"

                else:
                    # Ask the regulator (p53 checkpoint)
                    reg_client = OpenAIChatCompletionClient(model=MODEL, api_key=api_key)
                    approval = await evaluate_proposal(
                        original_task=task,
                        all_current_subtasks=all_subtask_texts,
                        current_depth=depth + 1,
                        current_total_agents=global_state.total_agents_spawned,
                        proposal=proposal,
                        model_client=reg_client,
                    )
                    await reg_client.close()
                    await emit({"type": "approval", "agent": name, **approval})

                    if approval["approved"]:
                        await log_emit(
                            f"Parent regulator approved '{name}'s proposal. "
                            f"Sub-agent '{name}' is spawning {proposed_n} children at depth {depth + 2}."
                        )
                        sub_result = await _run_lifecycle_core(
                            task=subtask,
                            depth=depth + 1,
                            global_state=global_state,
                            api_key=api_key,
                            name_prefix=name,
                            on_event=on_event,
                        )
                        result_text = sub_result["final_answer"]
                        child_nodes = sub_result["tree"].get("children", [])
                        decision_label = "propose_division_approved"
                    else:
                        await log_emit(
                            f"Parent regulator denied '{name}'s proposal because: "
                            f"{approval['reasoning']}. Sub-agent '{name}' will execute directly."
                        )
                        result_text = await run_subagent(agent, subtask)
                        decision_label = "execute_after_denial"
            else:
                result_text = await run_subagent(agent, subtask)
                decision_label = "execute"

        except Exception as exc:
            logger.error(f"Sub-agent '{name}' encountered an error: {exc}")
            await emit({"type": "log", "message": f"Sub-agent '{name}' encountered an error: {exc}."})
            result_text = f"[Error in {name}: {exc}]"
            decision_label = "error"

        await emit({"type": "result", "agent": name, "subtask": subtask, "result": result_text})
        await emit({"type": "log", "message": f"Collected result from '{name}': work complete."})

        return {
            "agent": name,
            "subtask": subtask,
            "effort_hint": effort_hint,
            "result": result_text,
            "client": client,
            "decision": decision_label,
            "children": child_nodes,
        }

    raw_results: list[dict] = []
    sub_results: list[dict] = []

    # as_completed streams results as each agent finishes — faster agents surface first
    for coro in asyncio.as_completed([_work_one(a, c, s, e) for a, c, s, e in spawned]):
        r = await coro
        raw_results.append(r)
        sub_results.append({"agent": r["agent"], "subtask": r["subtask"], "result": r["result"]})

    # ── DESTROY ─────────────────────────────────────────────────────────────
    await log_emit("Parent agent is beginning the destruction of sub-agents at this depth.")

    for r in raw_results:
        await destroy_subagent(r["agent"], r["client"], registry)
        remaining = registry.active_count()
        await emit({"type": "destroy", "agent": r["agent"], "remaining": remaining})
        await emit({
            "type": "log",
            "message": (
                f"Sub-agent '{r['agent']}' has been destroyed. "
                f"{remaining} sub-agent(s) remaining at depth {depth + 1}."
            ),
        })

    # ── SYNTHESIZE ───────────────────────────────────────────────────────────
    await log_emit("Parent agent is synthesizing all results into a final answer.")
    final_answer = await _synthesize(task, sub_results, api_key)

    await log_emit(
        f"Lifecycle at depth {depth} complete. "
        f"{len(sub_results)} sub-agent(s) contributed results."
    )

    tree = {
        "name": name_prefix or "root",
        "task": task,
        "depth": depth,
        "effort": effort,
        "final_answer": final_answer,
        "children": [
            {
                "name": r["agent"],
                "subtask": r["subtask"],
                "effort_hint": r["effort_hint"],
                "decision": r["decision"],
                "result": r["result"],
                "children": r.get("children", []),
            }
            for r in raw_results
        ],
    }

    return {
        "task": task,
        "effort": effort,
        "sub_tasks": [d["subtask"] for d in subtask_dicts],  # list[str] for backward compat
        "sub_results": sub_results,
        "final_answer": final_answer,
        "tree": tree,
    }


# ── Public entry points ────────────────────────────────────────────────────────

async def run_mitosys(task: str) -> dict:
    """Batch path: runs the full lifecycle and returns a complete result dict."""
    capture = _LogCapture()
    capture.setFormatter(
        logging.Formatter("%(asctime)s | MITOSYS | %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root_logger = logging.getLogger("mitosys")
    root_logger.addHandler(capture)

    try:
        api_key = os.environ["OPENAI_API_KEY"]
        logger.info(f"Parent agent received the task: '{task}'")

        global_state = GlobalState()
        result = await _run_lifecycle_core(task, 0, global_state, api_key, "", None)

        msg = (
            f"Mitosys lifecycle complete. Final tree depth: {global_state.max_depth_reached}. "
            f"Total agents spawned: {global_state.total_agents_spawned}."
        )
        logger.info(msg)

        return {
            "task": task,
            "effort": result["effort"],
            "sub_tasks": result["sub_tasks"],
            "sub_results": result["sub_results"],
            "final_answer": result["final_answer"],
            "tree": result["tree"],
            "log": list(capture.lines),
        }
    finally:
        root_logger.removeHandler(capture)


async def run_mitosys_stream(task: str):
    """
    Streaming path: async generator that yields SSE event dicts as they happen.

    Architecture: _run_lifecycle_core runs as a background asyncio Task and pushes
    events into a Queue. This generator drains the Queue and yields each event
    immediately, giving real-time streaming. A _Sentinel object signals completion.
    """
    api_key = os.environ["OPENAI_API_KEY"]
    global_state = GlobalState()
    queue: asyncio.Queue = asyncio.Queue()

    msg = f"Parent agent received the task: '{task}'"
    logger.info(msg)
    yield {"type": "log", "message": msg}

    async def push_event(event: dict):
        await queue.put(event)

    async def _run_and_signal():
        try:
            result = await _run_lifecycle_core(task, 0, global_state, api_key, "", push_event)
            await queue.put(_Sentinel(result=result))
        except Exception as exc:
            await queue.put(_Sentinel(error=str(exc)))

    runner = asyncio.create_task(_run_and_signal())

    result = None
    while True:
        item = await queue.get()
        if isinstance(item, _Sentinel):
            if item.error:
                raise RuntimeError(f"Mitosys lifecycle error: {item.error}")
            result = item.result
            break
        yield item

    await runner

    if result:
        yield {"type": "tree", "tree": result["tree"]}
        yield {"type": "final", "final_answer": result["final_answer"]}

    msg = (
        f"Mitosys lifecycle complete. Final tree depth: {global_state.max_depth_reached}. "
        f"Total agents spawned: {global_state.total_agents_spawned}."
    )
    logger.info(msg)
    yield {"type": "log", "message": msg}
    yield {"type": "done"}
