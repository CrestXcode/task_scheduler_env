---
title: Task Scheduler Env
emoji: 📋
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - productivity
  - scheduling
---

# Task Scheduler Environment

A real-world productivity RL environment where an AI agent learns to prioritise and complete workplace tasks before their deadlines.

Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — the open-source framework by Meta & Hugging Face for standardised RL environments.

## Overview

The agent receives a board of workplace tasks, each with a name, priority level, effort requirement, and deadline. At each step, the agent picks one task to work on. The environment rewards smart prioritisation and penalises missed deadlines.

**3 difficulty levels:**
- **Easy** — 5 tasks, generous deadlines, effort=1 each
- **Medium** — 8 tasks, tighter deadlines, varying effort (1-3 steps)
- **Hard** — 10 tasks, tight deadlines, urgent task injected at step 3

## Quick Start
```python
from openenv.core import GenericEnvClient

with GenericEnvClient(base_url="https://kashish014-task-scheduler-env.hf.space").sync() as env:
    result = env.reset(difficulty="easy")  # easy, medium, or hard
    print(result.observation)

    result = env.step({"task_id": 0})
    print(result.observation)
    print(result.reward)
```

## Action Space

```python
class TaskSchedulerAction(Action):
    task_id: int  # ID of the task to work on this step
```

## Observation Space

```python
class TaskSchedulerObservation(Observation):
    done: bool                  # Is the episode over?
    reward: Optional[float]     # Reward this step
    current_step: int           # Current time step
    tasks: List[dict]           # All tasks and their status
    message: str                # Feedback message
    score: float                # Running score 0.0-1.0
```

Each task in the list looks like:
```json
{
    "task_id": 0,
    "name": "Reply to emails",
    "priority": "high",
    "effort": 1,
    "deadline": 4,
    "completed": false
}
```

## Tasks

### Easy — 5 tasks, generous deadlines
All tasks have effort=1 (complete in one step). Deadlines are generous.
Agent must learn to prioritise high priority tasks first.

### Medium — 8 tasks, tighter deadlines
Tasks have varying effort (1-3 steps). Deadlines are tighter.
Agent must balance effort vs deadline vs priority.

### Hard — 10 tasks + urgent injection
Starts with 10 tasks. At step 3, an urgent high-priority task is injected
with a short deadline. Agent must dynamically reprioritise.

## Reward Function

| Event | Reward |
|---|---|
| Complete task on time | +1.0 |
| Progress on task (partial) | +0.1 per step |
| Working on high priority task | +0.1 bonus |
| Complete task late | +0.3 |
| Miss deadline | -0.3 |
| Ignore urgent task (deadline ≤ 2 steps) | -0.1 per step |
| Invalid task_id | -0.2 |

Final score = tasks completed / total tasks (0.0 to 1.0)

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks + action schema |
| `/grader` | GET | Grader info and criteria |
| `/baseline` | POST | Run baseline inference |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |

## Baseline Results

Run with Groq LLaMA 3.1 8B Instant as the agent:

| Difficulty | Success | Steps |
|---|---|---|
| Easy | true | 5 |
| Medium | false | 9 |
| Hard | false | 6 |

## Setup & Local Development

### Requirements
- Python 3.10+
- Docker
- openenv-core

### Run locally

```bash
git clone https://github.com/CrestXCode/task_scheduler_env
cd task_scheduler_env
pip install openenv-core
uv run server
```

### Run with Docker

```bash
docker build -t task-scheduler-env -f server/Dockerfile .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key task-scheduler-env
```

## Project Structure

```
task_scheduler/
├── models.py                          # Pydantic Action, Observation types
├── client.py                          # TaskSchedulerEnv client
├── inference.py                       # Baseline inference script
├── openenv.yaml                       # OpenEnv manifest
├── Dockerfile                         # Container definition (root, for HuggingFace)
├── README.md                          # This file
└── server/
    ├── task_scheduler_environment.py  # Core game logic
    ├── app.py                         # FastAPI server + endpoints
    └── Dockerfile                     # Container definition (local)
```

## Environment Motivation

Task prioritisation is a genuine challenge for AI agents in productivity tools,
project management systems, and autonomous assistants. This environment provides
a clean, standardised training ground for agents to learn deadline-aware,
priority-sensitive scheduling behaviour — directly applicable to real-world
AI assistant development.