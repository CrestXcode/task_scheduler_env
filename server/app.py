import sys
import os
import uvicorn

# Handle all possible import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from fastapi.responses import JSONResponse

try:
    from task_scheduler.models import TaskSchedulerAction, TaskSchedulerObservation
    from task_scheduler.server.task_scheduler_environment import TaskSchedulerEnvironment
except ImportError:
    try:
        from models import TaskSchedulerAction, TaskSchedulerObservation
        from server.task_scheduler_environment import TaskSchedulerEnvironment
    except ImportError:
        from task_scheduler.models import TaskSchedulerAction, TaskSchedulerObservation
        from task_scheduler.server.task_scheduler_environment import TaskSchedulerEnvironment

_difficulty_store = {"current": "easy"}

app = create_app(
    TaskSchedulerEnvironment,
    TaskSchedulerAction,
    TaskSchedulerObservation,
    env_name="task_scheduler",
    max_concurrent_envs=1,
)


async def set_difficulty(payload: dict):
    from server.task_scheduler_environment import _difficulty_store
    _difficulty_store["current"] = payload.get("difficulty", "easy")
    return JSONResponse({"difficulty": _difficulty_store["current"]})

app.add_api_route("/set_difficulty", set_difficulty, methods=["POST"])


async def get_tasks():
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


async def get_grader():
    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = TaskSchedulerEnvironment()
            env.reset(difficulty=difficulty)
            
            # Run a smart heuristic agent
            for _ in range(20):
                incomplete = [t for t in env._tasks if not t.completed]
                if not incomplete:
                    break
                # Pick highest priority, closest deadline
                task = sorted(
                    incomplete,
                    key=lambda t: (
                        {"high": 0, "medium": 1, "low": 2}[t.priority],
                        t.deadline
                    )
                )[0]
                from models import TaskSchedulerAction
                action = TaskSchedulerAction(task_id=task.task_id)
                result = env.step(action)
                if result.done:
                    break
            
            score = env.grader()
            # CLAMP SCORES - strictly between 0 and 1 (not 0.0, not 1.0)
            if score >= 0.99:
                score = 0.98
            if score <= 0.01:
                score = 0.02
            
            results[difficulty] = {
                "score": score,
                "tasks_completed": env._tasks_completed,
                "total_tasks": len(env._tasks),
            }
        except Exception as e:
            results[difficulty] = {"score": 0.02, "error": str(e)}

    return JSONResponse({
        "grader_results": results,
        "score_range": "0.02 to 0.98",
        "description": "Score = tasks completed / total tasks (clamped to 0.02-0.98 range)",
    })


async def run_baseline():
    import subprocess
    import json
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Parse the output to extract scores
            lines = result.stdout.strip().split('\n')
            scores = {"easy": 0.0, "medium": 0.0, "hard": 0.0}
            current_task = None
            for line in lines:
                if line.startswith("[START]"):
                    if "easy" in line:
                        current_task = "easy"
                    elif "medium" in line:
                        current_task = "medium"
                    elif "hard" in line:
                        current_task = "hard"
                elif line.startswith("[END]") and current_task:
                    if "success=true" in line:
                        scores[current_task] = 0.98
                    else:
                        scores[current_task] = 0.02
                    current_task = None
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


app.add_api_route("/tasks", get_tasks, methods=["GET"])
app.add_api_route("/grader", get_grader, methods=["GET"])
app.add_api_route("/baseline", run_baseline, methods=["POST"])


def main(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()