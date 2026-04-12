import os
import json
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from openenv.core import GenericEnvClient

load_dotenv()

API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

BASE_URL  = os.getenv("TASK_SCHEDULER_URL", "http://localhost:8000")
MAX_STEPS = 20

SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    {"name": "easy-scheduling",   "difficulty": "easy"},
    {"name": "medium-scheduling", "difficulty": "medium"},
    {"name": "hard-scheduling",   "difficulty": "hard"},
]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


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


def get_action(obs_dict: dict, incomplete: list, last=None) -> tuple:
    prompt = f"""You are an AI agent managing workplace tasks efficiently.
Step: {obs_dict.get('current_step', 0)} / {MAX_STEPS}

Incomplete tasks:
{json.dumps(incomplete, indent=2)}

Strategy:
- HIGH priority tasks first
- Among same priority, pick closest deadline
- If work_progress > 0, prefer finishing that task first

Reply with ONLY the integer task_id. Nothing else."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw     = (completion.choices[0].message.content or "").strip()
        digits  = ''.join(filter(str.isdigit, raw))
        task_id = int(digits) if digits else incomplete[0]["task_id"]
        valid_ids = [t["task_id"] for t in incomplete]
        return (task_id if task_id in valid_ids else incomplete[0]["task_id"]), None
    except Exception as e:
        return heuristic(obs_dict, last), str(e)[:50]


def run_episode(task: dict) -> None:
    task_name  = task["name"]
    difficulty = task["difficulty"]

    rewards     = []
    steps_taken = 0
    score       = 0.50
    success     = False
    last        = None
    obs         = None

    log_start(task=task_name, env="task-scheduler", model=MODEL_NAME)

    try:
        with GenericEnvClient(base_url=BASE_URL).sync() as env:
            result = env.reset(difficulty=difficulty)
            obs    = result.observation
            done   = result.done

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                obs_dict   = obs if isinstance(obs, dict) else vars(obs)
                tasks_all  = obs_dict.get("tasks", [])
                incomplete = [t for t in tasks_all if not t.get("completed")]

                if not incomplete:
                    success = True
                    break

                task_id, error = get_action(obs_dict, incomplete, last)
                last           = task_id
                action_str     = f"task_id={task_id}"

                result = env.step({"task_id": task_id})

                raw_reward = result.reward if result.reward is not None else 0.50
                reward     = float(raw_reward)
                # clamp strictly between 0 and 1
                reward     = round(min(max(reward, 0.01), 0.99), 2)

                rewards.append(reward)
                steps_taken = step
                obs         = result.observation
                done        = result.done

                log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                if done:
                    break

        obs_dict = obs if isinstance(obs, dict) else (vars(obs) if obs else {})
        raw_score = float(obs_dict.get("score", 0.50))
        score     = round(min(max(raw_score, 0.01), 0.99), 2)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        score   = 0.50
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    for task in TASKS:
        run_episode(task)


if __name__ == "__main__":
    main()