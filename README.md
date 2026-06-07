# 🧬 Mitosys

**Dynamic agent spawning, execution, and destruction — inspired by biological cell division.**

Mitosys is an open-source AI orchestration backend that takes any task, divides it into independent sub-tasks, spawns a dedicated AI agent for each one, runs them concurrently, collects their results, synthesizes a final answer, and then destroys every sub-agent — cleanly, completely, and with full lifecycle logs.

Built on [AutoGen](https://microsoft.github.io/autogen/stable/) and deployed on HuggingFace Spaces.

---

## 🔬 The Mitosis Metaphor

In biology, a cell divides to handle more work — each daughter cell does its job and dies. Mitosys applies this exact pattern to AI agents:

```
Task In
  │
  ├── Sub-agent 1 born → works → destroyed
  ├── Sub-agent 2 born → works → destroyed
  ├── Sub-agent 3 born → works → destroyed
  │
  └── Parent synthesizes all results → Final Answer Out
```

Every agent has a purpose. Every agent has an end.

---

## ⚡ Lifecycle — 7 Stages

| Stage | What Happens |
|---|---|
| **1. Receive** | Parent orchestrator receives the task |
| **2. Divide** | LLM planner decomposes the task into parallel-safe sub-tasks |
| **3. Spawn** | One sub-agent is born per sub-task, each with its own model client |
| **4. Execute** | All sub-agents work concurrently on their assigned sub-tasks |
| **5. Collect** | Results flow back to the parent orchestrator |
| **6. Destroy** | Each sub-agent is terminated and removed from the registry |
| **7. Return** | Parent synthesizes a final answer and returns it to the caller |

---

## 🏗️ Project Structure

```
mitosys-backend/
├── app/
│   ├── main.py          # FastAPI app and endpoints
│   ├── orchestrator.py  # 7-stage lifecycle manager
│   ├── divider.py       # LLM-based task decomposition
│   ├── subagent.py      # spawn, run, and destroy sub-agents
│   ├── registry.py      # tracks active sub-agents
│   └── logger.py        # clean, readable lifecycle logs
├── requirements.txt
├── run.sh
└── .env                 # OPENAI_API_KEY goes here (never commit this)
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/NorthCommits/mitosys-backend.git
cd mitosys-backend
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Add your OpenAI API key**
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

**4. Start the server**
```bash
./run.sh
```

Server runs at `http://localhost:8000`

---

## 📡 API Endpoints

### `POST /run`
Runs the full Mitosys lifecycle on a given task.

**Request**
```json
{
  "task": "Write a short report on the benefits of green tea."
}
```

**Response**
```json
{
  "task": "Write a short report on the benefits of green tea.",
  "sub_tasks": [
    "Research and write about the antioxidant properties of green tea.",
    "Research and write about green tea's effects on heart health.",
    "Research and write about green tea's role in weight management."
  ],
  "sub_results": [
    {
      "agent": "subagent_1",
      "subtask": "Research and write about the antioxidant properties of green tea.",
      "result": "..."
    }
  ],
  "final_answer": "...",
  "log": [
    "2026-06-07 13:33:45 | MITOSYS | Parent agent received the task.",
    "2026-06-07 13:33:46 | MITOSYS | Parent divided the task into 3 sub-tasks.",
    "2026-06-07 13:33:46 | MITOSYS | Sub-agent 'subagent_1' has been born.",
    "2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_1' has been destroyed. 2 sub-agent(s) remaining.",
    "2026-06-07 13:33:57 | MITOSYS | Mitosys lifecycle complete. Returning the final answer to the caller."
  ]
}
```

**Try it**
```bash
curl -s -X POST https://your-space.hf.space/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a short report on the benefits of green tea."}' \
  | python3 -m json.tool
```

---

### `GET /health`
```json
{ "status": "ok" }
```

### `GET /`
Returns a description of what Mitosys is.

---

## 📋 Sample Lifecycle Log

This is what a real Mitosys run looks like in the server terminal — clean, readable, no JSON noise:

```
2026-06-07 13:33:45 | MITOSYS | Parent agent received the task: 'Write a short report on the benefits of green tea.'
2026-06-07 13:33:45 | MITOSYS | Parent agent is consulting the LLM to divide the task into sub-tasks.
2026-06-07 13:33:46 | MITOSYS | Parent divided the task into 5 sub-tasks.
2026-06-07 13:33:46 | MITOSYS | Parent agent is spawning 5 sub-agent(s).
2026-06-07 13:33:46 | MITOSYS | Sub-agent 'subagent_1' has been born and assigned its sub-task.
2026-06-07 13:33:46 | MITOSYS | Sub-agent 'subagent_2' has been born and assigned its sub-task.
2026-06-07 13:33:46 | MITOSYS | Sub-agent 'subagent_3' has been born and assigned its sub-task.
2026-06-07 13:33:46 | MITOSYS | Sub-agent 'subagent_4' has been born and assigned its sub-task.
2026-06-07 13:33:46 | MITOSYS | Sub-agent 'subagent_5' has been born and assigned its sub-task.
2026-06-07 13:33:46 | MITOSYS | All 5 sub-agent(s) are now working concurrently on their sub-tasks.
2026-06-07 13:33:47 | MITOSYS | Sub-agent 'subagent_5' has finished its sub-task and returned its result.
2026-06-07 13:33:50 | MITOSYS | Sub-agent 'subagent_2' has finished its sub-task and returned its result.
2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_1' has finished its sub-task and returned its result.
2026-06-07 13:33:52 | MITOSYS | Parent agent is beginning the destruction of all sub-agents.
2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_1' has been destroyed. 4 sub-agent(s) remaining.
2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_2' has been destroyed. 3 sub-agent(s) remaining.
2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_3' has been destroyed. 2 sub-agent(s) remaining.
2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_4' has been destroyed. 1 sub-agent(s) remaining.
2026-06-07 13:33:52 | MITOSYS | Sub-agent 'subagent_5' has been destroyed. 0 sub-agent(s) remaining.
2026-06-07 13:33:52 | MITOSYS | Parent agent is synthesizing all results into a final answer.
2026-06-07 13:33:57 | MITOSYS | Mitosys lifecycle complete. Returning the final answer to the caller.
```

---

## 🤝 Framework Agnostic Design

Mitosys is built on AutoGen but the orchestration pattern is framework-agnostic by design. The core lifecycle — divide, spawn, execute, collect, destroy — can be adapted to LangGraph, CrewAI, or any other multi-agent framework. Adapter support is on the roadmap.

---

## 🗺️ Roadmap

- [x] Core 7-stage lifecycle
- [x] Parallel sub-agent execution
- [x] Clean lifecycle logging
- [x] FastAPI backend
- [x] HuggingFace Spaces deployment
- [ ] Maze exploration demo (parallel BFS via sub-agents)
- [ ] CLI package (`pip install mitosys-cli`)
- [ ] Sequential mode (agents that share context)
- [ ] Framework adapters (LangGraph, CrewAI)
- [ ] Rate limiting and token auth

---

## 🛡️ Notes

- This backend uses a shared OpenAI API key hosted as a HuggingFace Secret. Please be respectful of usage.
- Rate limiting will be added in an upcoming release.
- Never commit your `.env` file. Add it to `.gitignore`.

---

## 👤 Author

Built by [Swapnil Bhattacharya](https://github.com/swapnilbhattacharya) — AI/GenAI Engineer, open-source builder.

Part of the [NorthCommits](https://github.com/NorthCommits) open-source organisation.

---

## 📄 License

MIT