import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.logger import get_logger
from app.orchestrator import run_mitosys

load_dotenv()

logger = get_logger("mitosys.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Mitosys API is starting up.")
    yield
    logger.info("Mitosys API is shutting down.")


app = FastAPI(title="Mitosys", lifespan=lifespan)


class RunRequest(BaseModel):
    task: str


@app.get("/")
async def root():
    return {
        "name": "Mitosys",
        "description": (
            "A multi-agent backend inspired by biological cell division. "
            "A parent orchestrator receives a task, divides it into sub-tasks, "
            "spawns one AssistantAgent per sub-task, runs them concurrently, "
            "collects their results, synthesizes a final answer, then destroys "
            "every sub-agent — cleanly logging each stage of the lifecycle."
        ),
        "endpoints": {
            "POST /run": "Submit a task and receive the full mitosis lifecycle result.",
            "GET /health": "Health check.",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run")
async def run(request: RunRequest):
    task = request.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="The 'task' field must not be empty.")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set. Please add it to your .env file.",
        )

    logger.info(f"API received a new run request.")
    try:
        result = await run_mitosys(task)
        return result
    except Exception as exc:
        logger.error(f"An error occurred during the Mitosys run: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
