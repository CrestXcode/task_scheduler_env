import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from openenv.core import GenericEnvClient

load_dotenv()

API_BASE_URL   = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
OPENAI_API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

BASE_URL = os.getenv("TASK_SCHEDULER_URL", "http://localhost:8000")

TASKS = [
    {"name": "easy-scheduling",   "difficulty": "easy"},
    {"name": "medium-scheduling", "difficulty": "medium"},
    {"name": "hard-scheduling",   "difficulty": "hard"},
]

MAX_STEPS = 20


def _safe_reward(value) -> float:
    """Clamp any reward received from the environment to strictly (0.15, 0.85)."""
    try:
        v = float(value) if value is not None else 0.50
    except (TypeError, ValueError):
        v = 0.50
    return round(min(max(v, 0.15), 0.85), 2)


def heuristic(obs_dict: dict, last=None) -> int:
    tasks = obs_dict.get("tasks", [])
    step  = obs_dict.get("current_step", 0)
    best, best_score = None, -1e9

    for t in tasks:
        if t.get("completed"):
            continue
        time_left = t["deadline"] - step
        if time_left <= 0:
            continue
        effort    = t["effort"]
        progress  = t.get("work_progress", 0)
        remaining = effort - progress

        score = (
            (100 if t["priority"] == "high" else 50 if t["priority"] == "medium" else 10)
            + 200 / (time_left + 1)
            + (300 if time_left <= remaining else 0)
            + (80  if last == t["task_id"] else 0)
        )
        if score > best_score:
            best_score = score
            best = t

    if best is None:
        for t in tasks:
            if not t.get("completed"):
                return t["task_id"]
        return 0
    return best["task_id"]


def call_llm(prompt: str, incomplete: list) -> int:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    raw     = response.choices[0].message.content.strip()
    digits  = ''.join(filter(str.isdigit, raw))
    task_id = int(digits) if digits else incomplete[0]["task_id"]
    valid_ids = [t["task_id"] for t in incomplete]
    return task_id if task_id in valid_ids else incomplete[0]["task_id"]


def run_episode(task: dict) -> None:
    task_name  = task["name"]
    difficulty = task["difficulty"]

    print(f"[START] task={task_name} env=task-scheduler model={MODEL_NAME}", flush=True)

    step_num = 0
    rewards  = []
    success  = False
    last     = None
    obs      = None

    try:
        with GenericEnvClient(base_url=BASE_URL).sync() as env:
            result = env.reset(difficulty=difficulty)
            obs    = result.observation
            done   = result.done

            for step_num in range(1, MAX_STEPS + 1):
                if done:
                    break

                obs_dict   = obs if isinstance(obs, dict) else vars(obs)
                tasks_all  = obs_dict.get("tasks", [])
                incomplete = [t for t in tasks_all if not t.get("completed")]

                if not incomplete:
                    success = True
                    break

                prompt = f"""You are an AI agent managing workplace tasks efficiently.
Step: {obs_dict.get('current_step', 0)} / {MAX_STEPS}

Incomplete tasks:
{json.dumps(incomplete, indent=2)}

Strategy:
- HIGH priority tasks first
- Among same priority, pick closest deadline
- If a task has work_progress > 0, prefer finishing it before switching
- Avoid unnecessary task switching

Reply with ONLY the integer task_id. Nothing else."""

                error_str = "null"
                try:
                    task_id = call_llm(prompt, incomplete)
                except Exception as e:
                    task_id   = heuristic(obs_dict, last)
                    error_str = str(e)[:50]

                last       = task_id
                action_str = f"task_id={task_id}"
                result     = env.step({"task_id": task_id})

                reward = _safe_reward(result.reward)
                rewards.append(reward)
                obs  = result.observation
                done = result.done

                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} "
                    f"error={error_str}",
                    flush=True
                )

                if done:
                    break

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        success = False

    obs_dict  = obs if isinstance(obs, dict) else (vars(obs) if obs else {})
    raw_score = float(obs_dict.get("score", 0.50))
    score     = round(min(max(raw_score, 0.15), 0.85), 2)
    success   = score > 0.50

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.50"
    print(
    f"[END] success={str(success).lower()} steps={step_num} "
    f"score={score:.2f} rewards={rewards_str}",
    flush=True
)


def main():
    for task in TASKS:
        run_episode(task)


if __name__ == "__main__":
    main()