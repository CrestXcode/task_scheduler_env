from openenv.core.env_server.http_server import create_app

try:
    from ..models import TaskSchedulerAction, TaskSchedulerObservation
    from .task_scheduler_environment import TaskSchedulerEnvironment
except ModuleNotFoundError:
    from models import TaskSchedulerAction, TaskSchedulerObservation
    from server.task_scheduler_environment import TaskSchedulerEnvironment


app = create_app(
    TaskSchedulerEnvironment,
    TaskSchedulerAction,
    TaskSchedulerObservation,
    env_name="task_scheduler",
    max_concurrent_envs=1,
)


# ─── Required hackathon endpoints ────────────────────────────────────────────

from fastapi.responses import JSONResponse


@app.get("/tasks")
async def get_tasks():
    """Returns list of tasks and action schema."""
    return JSONResponse({
        "tasks": [
            {
                "id": "easy",
                "name": "Easy Scheduling",
                "description": "Schedule 5 tasks with generous deadlines. All effort=1.",
                "difficulty": "easy",
                "max_steps": 20,
            },
            {
                "id": "medium",
                "name": "Medium Scheduling",
                "description": "Schedule 8 tasks with tighter deadlines and varying effort.",
                "difficulty": "medium",
                "max_steps": 20,
            },
            {
                "id": "hard",
                "name": "Hard Scheduling",
                "description": "Schedule 10 tasks. Urgent task injected at step 3.",
                "difficulty": "hard",
                "max_steps": 20,
            },
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to work on this step",
                }
            },
            "required": ["task_id"],
        },
    })


@app.get("/grader")
async def get_grader():
    """Returns grader score info and criteria."""
    return JSONResponse({
        "score_range": "0.0 to 1.0",
        "description": "Score based on tasks completed on time vs total tasks",
        "criteria": {
            "1.0": "All tasks completed on time",
            "0.5": "Half of tasks completed",
            "0.0": "No tasks completed",
        },
        "partial_credit": {
            "on_time_completion": "+1.0 reward",
            "late_completion": "+0.3 reward",
            "missed_deadline": "-0.3 reward",
            "ignoring_urgent": "-0.1 per step",
            "invalid_action": "-0.2 reward",
        },
    })


@app.post("/baseline")
async def run_baseline():
    """Triggers Gemini baseline inference and returns scores for all 3 tasks."""
    import subprocess
    import sys
    import json

    try:
        result = subprocess.run(
            [sys.executable, "baseline.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            scores = json.loads(result.stdout)
            return JSONResponse({"status": "success", "scores": scores})
        else:
            return JSONResponse(
                {"status": "error", "error": result.stderr},
                status_code=500,
            )
    except Exception as e:
        return JSONResponse(
            {"status": "error", "error": str(e)},
            status_code=500,
        )


# ─── Server entrypoint ────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)