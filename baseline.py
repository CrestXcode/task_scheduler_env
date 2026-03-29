import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from openenv.core import GenericEnvClient

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

BASE_URL = "http://localhost:8000"
DIFFICULTIES = ["easy", "medium", "hard"]


def run_episode(difficulty: str) -> float:
    """Run one full episode with Groq LLM as the agent. Returns final score 0.0-1.0."""

    with GenericEnvClient(base_url=BASE_URL).sync() as env:
        result = env.reset()

        for step in range(20):
            obs = result.observation
            done = result.done

            if done:
                break

            tasks = obs.get("tasks", [])
            incomplete = [t for t in tasks if not t["completed"]]

            if not incomplete:
                break

            prompt = f"""You are an AI productivity agent managing workplace tasks.
Your job is to pick which task to work on at each step.

Current step: {obs.get('current_step', 0)}
Last message: {obs.get('message', '')}
Current score: {obs.get('score', 0.0)}

Incomplete tasks:
{json.dumps(incomplete, indent=2)}

Rules for picking:
- Always prioritise HIGH priority tasks first
- Among same priority, pick the one with closest deadline
- deadline means the step number by which it must be done
- effort means how many steps it takes to complete

Reply with ONLY a single integer — the task_id to work on.
Do not explain. Just the number."""

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
                digits = ''.join(filter(str.isdigit, raw))
                task_id = int(digits) if digits else incomplete[0]["task_id"]
            except Exception:
                task_id = incomplete[0]["task_id"]

            valid_ids = [t["task_id"] for t in incomplete]
            if task_id not in valid_ids:
                task_id = incomplete[0]["task_id"]

            result = env.step({"task_id": task_id})

        return result.observation.get("score", 0.0)


def main():
    scores = {}
    for difficulty in DIFFICULTIES:
        try:
            score = run_episode(difficulty)
            scores[difficulty] = round(score, 2)
        except Exception as e:
            scores[difficulty] = 0.0

    print(json.dumps(scores))


if __name__ == "__main__":
    main()