import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from openenv.core import GenericEnvClient

load_dotenv()

# Use defaults so validator doesn't crash
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Initialize client (will use their values when injected)
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# Use LIVE HuggingFace URL
BASE_URL = "https://kashish014-task-scheduler-env.hf.space"
TASKS = [
    {"name": "easy-scheduling", "difficulty": "easy"},
    {"name": "medium-scheduling", "difficulty": "medium"},
    {"name": "hard-scheduling", "difficulty": "hard"},
]

def run_episode(task: dict) -> None:
    task_name = task["name"]
    difficulty = task["difficulty"]

    print(f"[START] task={task_name} env=task-scheduler model={MODEL_NAME}", flush=True)

    step_num = 0
    rewards = []
    success = False

    try:
        with GenericEnvClient(base_url=BASE_URL).sync() as env:
            result = env.reset(difficulty=difficulty)
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

Reply with ONLY a single integer - the task_id to work on."""

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
                    task_id = incomplete[0]["task_id"]
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

                print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}", flush=True)

                if done:
                    success = obs.get("score", 0.0) >= 0.5
                    break

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str if rewards_str else '0.00'}", flush=True)

def main():
    for task in TASKS:
        run_episode(task)

if __name__ == "__main__":
    main()