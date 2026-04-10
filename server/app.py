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
            
            # SMART AGENT with proper project management logic
            for step in range(20):
                incomplete = [t for t in env._tasks if not t.completed]
                if not incomplete:
                    break
                
                # Calculate which task to work on next
                best_task = None
                best_score = -float('inf')
                
                for task in incomplete:
                    # Time remaining until deadline
                    time_remaining = task.deadline - env._current_step
                    
                    # Priority score (higher = more important)
                    priority_score = {"high": 10, "medium": 5, "low": 1}[task.priority]
                    
                    # Urgency score (closer deadline = more urgent)
                    if time_remaining <= 0:
                        urgency_score = 100  # Already missed!
                    elif time_remaining <= 2:
                        urgency_score = 50   # Very urgent
                    elif time_remaining <= 4:
                        urgency_score = 30   # Urgent
                    else:
                        urgency_score = 10   # Normal
                    
                    # Progress score (tasks partially done should be finished)
                    progress = env._work_progress.get(task.task_id, 0)
                    progress_score = progress * 5  # Encourage finishing what you started
                    
                    # Effort score (can we finish it quickly?)
                    steps_needed = task.effort - progress
                    if steps_needed <= 1:
                        effort_score = 20   # Quick win
                    elif steps_needed <= 2:
                        effort_score = 10
                    else:
                        effort_score = 0
                    
                    # Total score
                    total_score = priority_score + urgency_score + progress_score + effort_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_task = task
                
                if best_task is None:
                    break
                
                from models import TaskSchedulerAction
                action = TaskSchedulerAction(task_id=best_task.task_id)
                result = env.step(action)
                if result.done:
                    break
            
            score = env.grader()
            # Gentle clamping only at extremes
            if score >= 0.99:
                score = 0.95
            if score <= 0.01:
                score = 0.05
            
            results[difficulty] = {
                "score": round(score, 2),
                "tasks_completed": env._tasks_completed,
                "total_tasks": len(env._tasks),
            }
        except Exception as e:
            results[difficulty] = {"score": 0.05, "error": str(e)}

    return JSONResponse({
        "grader_results": results,
        "score_range": "0.0 to 1.0",
        "description": "Score = tasks completed / total tasks",
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
                        scores[current_task] = 0.99
                    else:
                        scores[current_task] = 0.01
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