import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from openenv.core import GenericEnvClient

load_dotenv()

# Using THEIR variable names exactly as they inject
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY") 
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Validate required vars
if not API_BASE_URL:
    raise ValueError("API_BASE_URL environment variable is required")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is required")

# Initialize client with THEIR variables
client = OpenAI(
    api_key=API_KEY,           # ← API_KEY, not OPENAI_API_KEY
    base_url=API_BASE_URL      # ← their proxy endpoint
)
TASKS = [
    {"name": "easy-scheduling",   "url": "https://kashish014-task-scheduler-env.hf.space", "difficulty": "easy"},
    {"name": "medium-scheduling", "url": "https://kashish014-task-scheduler-env.hf.space", "difficulty": "medium"},
    {"name": "hard-scheduling",   "url": "https://kashish014-task-scheduler-env.hf.space", "difficulty": "hard"},
]

def run_episode(task: dict) -> None:
    task_name = task["name"]
    base_url = task["url"]

    print(f"[START] task={task_name} env=task-scheduler model={MODEL_NAME}", flush=True)

    step_num = 0
    rewards = []
    success = False

    try:
        with GenericEnvClient(base_url=base_url).sync() as env:
            result = env.reset(difficulty=task["difficulty"])
            obs = result.observation
            done = result.done

            for step_num in range(1, 21):
                if done:
                    success = obs.get("score", 0.0) >= 0.5
                    break

                tasks = obs.get("tasks", [])
                incomplete = [t for t in tasks if not t["completed"]]

                if not incomplete:
                    success = True
                    break

                prompt = f"""You are an AI productivity agent managing workplace tasks.
Current step: {obs.get('current_step', 0)}
Last message: {obs.get('message', '')}

Incomplete tasks:
{json.dumps(incomplete, indent=2)}

Rules:
- Always prioritise HIGH priority tasks first
- Among same priority, pick the one with closest deadline
- deadline = step number by which task must be done
- effort = how many steps to complete

Reply with ONLY a single integer - the task_id to work on.
Do not explain. Just the number."""

                error_str = "null"
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        temperature=0.0,
                    )
                    raw = response.choices[0].message.content.strip()
                    digits = ''.join(filter(str.isdigit, raw))
                    task_id = int(digits) if digits else incomplete[0]["task_id"]
                except Exception as e:
                    # Fallback: pick highest priority, closest deadline
                    task_id = sorted(
                        incomplete,
                        key=lambda t: ({"high": 0, "medium": 1, "low": 2}[t["priority"]], t["deadline"])
                    )[0]["task_id"]
                    error_str = str(e)[:50]

                valid_ids = [t["task_id"] for t in incomplete]
                if task_id not in valid_ids:
                    task_id = incomplete[0]["task_id"]

                action_str = f"task_id={task_id}"
                result = env.step({"task_id": task_id})

                reward = result.reward if result.reward is not None else 0.0
                rewards.append(reward)
                obs = result.observation
                done = result.done

                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={str(done).lower()} "
                    f"error={error_str}",
                    flush=True
                )

                if done:
                    success = obs.get("score", 0.0) >= 0.5
                    break

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={step_num} "
        f"rewards={rewards_str if rewards_str else '0.00'}",
        flush=True
    )


def main():
    for task in TASKS:
        run_episode(task)


if __name__ == "__main__":
    main()