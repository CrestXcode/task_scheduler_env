import uuid
import json
import os
import tempfile
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TaskSchedulerAction, TaskSchedulerObservation, Task
except ImportError:
    from models import TaskSchedulerAction, TaskSchedulerObservation, Task


TASK_TEMPLATES = {
    "easy": [
        {"name": "Email",   "priority": "high",   "effort": 1, "deadline": 3},
        {"name": "Standup", "priority": "medium",  "effort": 1, "deadline": 4},
        {"name": "Review",  "priority": "high",    "effort": 1, "deadline": 5},
        {"name": "Board",   "priority": "low",     "effort": 1, "deadline": 6},
        {"name": "Docs",    "priority": "low",     "effort": 1, "deadline": 7},
        {"name": "Slack",   "priority": "medium",  "effort": 1, "deadline": 4},
    ],
    "medium": [
        {"name": "Tests",    "priority": "high",   "effort": 2, "deadline": 6},
        {"name": "Debug",    "priority": "high",   "effort": 2, "deadline": 5},
        {"name": "Review",   "priority": "high",   "effort": 2, "deadline": 7},
        {"name": "Docs",     "priority": "medium", "effort": 2, "deadline": 8},
        {"name": "Meeting",  "priority": "medium", "effort": 1, "deadline": 5},
        {"name": "Bugfix",   "priority": "high",   "effort": 1, "deadline": 4},
        {"name": "Optimize", "priority": "medium", "effort": 2, "deadline": 7},
    ],
    "hard": [
        {"name": "Migration",   "priority": "high",   "effort": 3, "deadline": 6},
        {"name": "Security",    "priority": "high",   "effort": 3, "deadline": 5},
        {"name": "Deploy",      "priority": "high",   "effort": 2, "deadline": 4},
        {"name": "Incident",    "priority": "high",   "effort": 2, "deadline": 6},
        {"name": "Performance", "priority": "medium", "effort": 3, "deadline": 8},
        {"name": "OAuth",       "priority": "high",   "effort": 3, "deadline": 7},
        {"name": "Load",        "priority": "medium", "effort": 2, "deadline": 6},
        {"name": "Hotfix",      "priority": "high",   "effort": 1, "deadline": 3},
        {"name": "Audit",       "priority": "high",   "effort": 2, "deadline": 5},
        {"name": "Client",      "priority": "high",   "effort": 2, "deadline": 4},
    ],
}

MAX_STEPS  = 20
STATE_FILE = os.path.join(tempfile.gettempdir(), "task_scheduler_state.json")


def _clip(value: float) -> float:
    """Clamp to (0.15, 0.85) — well inside (0, 1) exclusive."""
    return round(float(min(max(value, 0.16), 0.73)), 2)


_GLOBAL_ENV = None


def get_global_env():
    global _GLOBAL_ENV
    if _GLOBAL_ENV is None:
        _GLOBAL_ENV = TaskSchedulerEnvironment()
    return _GLOBAL_ENV


class TaskSchedulerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._state           = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._tasks           = []
        self._current_step    = 0
        self._tasks_completed = 0
        self._progress        = {}
        self._on_time         = set()
        self._difficulty      = "easy"

    def _build_tasks(self, difficulty):
        return [
            Task(i, t["name"], t["priority"], t["effort"], t["deadline"], False)
            for i, t in enumerate(TASK_TEMPLATES[difficulty])
        ]

    def _save_state(self):
        try:
            data = {
                "difficulty":      self._difficulty,
                "current_step":    self._current_step,
                "tasks_completed": self._tasks_completed,
                "progress":        self._progress,
                "on_time":         list(self._on_time),
                "episode_id":      self._state.episode_id,
                "step_count":      self._state.step_count,
                "tasks": [
                    {"task_id": t.task_id, "completed": t.completed}
                    for t in self._tasks
                ],
            }
            with open(STATE_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_state(self) -> bool:
        try:
            if not os.path.exists(STATE_FILE):
                return False
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            difficulty            = data.get("difficulty", "easy")
            self._difficulty      = difficulty
            self._tasks           = self._build_tasks(difficulty)
            completed_map         = {t["task_id"]: t["completed"] for t in data.get("tasks", [])}
            for t in self._tasks:
                t.completed = completed_map.get(t.task_id, False)
            self._current_step    = data.get("current_step", 0)
            self._tasks_completed = data.get("tasks_completed", 0)
            self._progress        = {int(k): v for k, v in data.get("progress", {}).items()}
            self._on_time         = set(data.get("on_time", []))
            self._state = State(
                episode_id=data.get("episode_id", str(uuid.uuid4())),
                step_count=data.get("step_count", 0),
            )
            return True
        except Exception:
            return False

    def reset(self, difficulty="easy", **kwargs):
        self._difficulty      = difficulty
        self._tasks           = self._build_tasks(difficulty)
        self._current_step    = 0
        self._tasks_completed = 0
        self._progress        = {t.task_id: 0 for t in self._tasks}
        self._on_time         = set()
        self._state           = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._save_state()

        return TaskSchedulerObservation(
            done=False,
            reward=0.50,
            current_step=0,
            tasks=[t.to_dict() for t in self._tasks],
            message="Episode started",
            score=0.50,
        )

    def grader(self) -> float:
        total = len(self._tasks)
        if total == 0:
            return 0.50
        completion = self._tasks_completed / total
        on_time    = len(self._on_time) / total
        raw        = 0.6 * completion + 0.4 * on_time
        # Maps [0, 1] → [0.20, 0.80] — mathematically impossible to hit 0 or 1
        return round(0.27 + (raw * 0.46), 2)

    def step(self, action: TaskSchedulerAction, **kwargs):

        self._current_step     += 1
        self._state.step_count += 1

        task = next(
            (t for t in self._tasks if t.task_id == action.task_id and not t.completed),
            None
        )

        if task:
            self._progress[task.task_id] = self._progress.get(task.task_id, 0) + 1
            if self._progress[task.task_id] >= task.effort:
                task.completed = True
                self._tasks_completed += 1
                if self._current_step <= task.deadline:
                    reward = 0.73
                    self._on_time.add(task.task_id)
                else:
                    reward = 0.28
            else:
                ratio  = self._progress[task.task_id] / task.effort
                reward = _clip(0.20 + 0.28 * ratio)
        else:
            reward = 0.16

        done = (
            self._current_step >= MAX_STEPS
            or all(t.completed for t in self._tasks)
        )

        self._save_state()

        return TaskSchedulerObservation(
            done=done,
            reward=_clip(reward),
            current_step=self._current_step,
            tasks=[t.to_dict() for t in self._tasks],
            message="ok",
            score=_clip(self.grader()),
        )

    @property
    def state(self):
        return self._state