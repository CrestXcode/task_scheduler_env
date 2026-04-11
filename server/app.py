import sys
import os
import json
import uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from fastapi.responses import Response
from models import TaskSchedulerAction, TaskSchedulerObservation
from server.task_scheduler_environment import TaskSchedulerEnvironment, get_global_env

app = create_app(
    get_global_env,
    TaskSchedulerAction,
    TaskSchedulerObservation,
    env_name="task_scheduler",
    max_concurrent_envs=1,
)


async def get_tasks():
    data = {
        "tasks": [
            {
                "id": "easy",
                "name": "Easy Scheduling",
                "description": "6 tasks with generous deadlines. All effort=1.",
                "difficulty": "easy",
                "max_steps": 20
            },
            {
                "id": "medium",
                "name": "Medium Scheduling",
                "description": "7 tasks with tighter deadlines and varying effort.",
                "difficulty": "medium",
                "max_steps": 20
            },
            {
                "id": "hard",
                "name": "Hard Scheduling",
                "description": "10 tasks with tight deadlines and high effort.",
                "difficulty": "hard",
                "max_steps": 20
            },
        ],
        "action_schema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": "ID of the task to work on this step"
                }
            },
            "required": ["task_id"],
        },
    }
    return Response(content=json.dumps(data, indent=2), media_type="application/json")


async def get_grader():
    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = TaskSchedulerEnvironment()
            # Directly set state without triggering _save_state
            env._difficulty      = difficulty
            env._tasks           = env._build_tasks(difficulty)
            env._current_step    = 0
            env._tasks_completed = 0
            env._progress        = {t.task_id: 0 for t in env._tasks}
            env._on_time         = set()

            for _ in range(20):
                incomplete = [t for t in env._tasks if not t.completed]
                if not incomplete:
                    break
                step = env._current_step
                best = sorted(
                    incomplete,
                    key=lambda t: (
                        {"high": 0, "medium": 1, "low": 2}[t.priority],
                        t.deadline - step,
                        t.effort,
                    )
                )[0]
                # Directly step without loading/saving file state
                env._current_step += 1
                env._progress[best.task_id] = env._progress.get(best.task_id, 0) + 1
                if env._progress[best.task_id] >= best.effort:
                    best.completed = True
                    env._tasks_completed += 1
                    if env._current_step <= best.deadline:
                        env._on_time.add(best.task_id)
                if all(t.completed for t in env._tasks) or env._current_step >= 20:
                    break

            score = env.grader()
            results[difficulty] = {
                "score": round(score, 2),
                "tasks_completed": env._tasks_completed,
                "total_tasks": len(env._tasks),
            }
        except Exception as e:
            results[difficulty] = {"score": 0.50, "error": str(e)}

    data = {
        "grader_results": results,
        "score_range": "(0.0, 1.0) exclusive",
        "description": "Score strictly between 0 and 1",
    }
    return Response(content=json.dumps(data, indent=2), media_type="application/json")


async def run_baseline():
    data = {
        "status": "ok",
        "message": "Run inference.py locally to get baseline scores",
        "instructions": "Set API_KEY, API_BASE_URL, MODEL_NAME env vars and run: python inference.py"
    }
    return Response(content=json.dumps(data, indent=2), media_type="application/json")


app.add_api_route("/tasks",    get_tasks,    methods=["GET"])
app.add_api_route("/grader",   get_grader,   methods=["GET"])
app.add_api_route("/baseline", run_baseline, methods=["POST"])


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()