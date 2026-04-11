import sys
import os
import uvicorn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from fastapi.responses import JSONResponse
from models import TaskSchedulerAction, TaskSchedulerObservation
from server.task_scheduler_environment import TaskSchedulerEnvironment

app = create_app(
    TaskSchedulerEnvironment,
    TaskSchedulerAction,
    TaskSchedulerObservation,
    env_name="task_scheduler",
    max_concurrent_envs=1,
)


async def get_tasks():
    return JSONResponse({
        "tasks": [
            {"id": "easy",   "name": "Easy Scheduling",   "description": "5 tasks, generous deadlines.", "difficulty": "easy",   "max_steps": 20},
            {"id": "medium", "name": "Medium Scheduling", "description": "7 tasks, tighter deadlines.",  "difficulty": "medium", "max_steps": 20},
            {"id": "hard",   "name": "Hard Scheduling",   "description": "10 tasks, tight deadlines.",   "difficulty": "hard",   "max_steps": 20},
        ],
        "action_schema": {
            "type": "object",
            "properties": {"task_id": {"type": "integer", "description": "ID of the task to work on"}},
            "required": ["task_id"],
        },
    })


async def get_grader():
    results = {}
    for difficulty in ["easy", "medium", "hard"]:
        try:
            env = TaskSchedulerEnvironment()
            env.reset(difficulty=difficulty)
            tasks = env._tasks[:]

            for _ in range(20):
                incomplete = [t for t in tasks if not t.completed]
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
                from models import TaskSchedulerAction
                result = env.step(TaskSchedulerAction(task_id=best.task_id))
                if result.done:
                    break

            score = env.grader()
            results[difficulty] = {
                  "score": round(score, 2),
                  "tasks_completed": env._tasks_completed,
                  "total_tasks": len(env._tasks),
   }
        except Exception as e:
            results[difficulty] = {"score": 0.5, "error": str(e)}

    return JSONResponse({
        "grader_results": results,
        "score_range": "(0.0, 1.0) exclusive",
        "description": "Score strictly between 0 and 1",
    })


async def run_baseline():
    return JSONResponse({
        "status": "ok",
        "message": "Run inference.py locally to get baseline scores",
    })


app.add_api_route("/tasks",    get_tasks,    methods=["GET"])
app.add_api_route("/grader",   get_grader,   methods=["GET"])
app.add_api_route("/baseline", run_baseline, methods=["POST"])


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()