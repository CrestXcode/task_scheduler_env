import uuid
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

MAX_STEPS = 20


def _clip(value: float) -> float:
    return float(min(max(value, 0.06), 0.94))


class TaskSchedulerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._tasks = []
        self._current_step = 0
        self._tasks_completed = 0
        self._progress = {}
        self._on_time = set()

    def _build_tasks(self, difficulty):
        return [
            Task(i, t["name"], t["priority"], t["effort"], t["deadline"], False)
            for i, t in enumerate(TASK_TEMPLATES[difficulty])
        ]

    def reset(self, difficulty="easy"):
        self._tasks = self._build_tasks(difficulty)
        self._current_step = 0
        self._tasks_completed = 0
        self._progress = {t.task_id: 0 for t in self._tasks}
        self._on_time = set()
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

        return TaskSchedulerObservation(
            done=False,
            reward=0.05,
            current_step=0,
            tasks=[t.to_dict() for t in self._tasks],
            message="Episode started",
            score=0.05,
        )


    def grader(self) -> float:
       total = len(self._tasks)
       if total == 0:
          return 0.50
       completion = self._tasks_completed / total
       on_time = len(self._on_time) / total
       raw = 0.6 * completion + 0.4 * on_time
       # Add small noise to prevent exact boundary values
       result = 0.06 + (raw * 0.88)
       return round(result, 2)

    def step(self, action: TaskSchedulerAction):
        self._current_step += 1
        self._state.step_count += 1

        task = next(
            (t for t in self._tasks if t.task_id == action.task_id and not t.completed),
            None
        )

        if task:
            self._progress[task.task_id] += 1
            if self._progress[task.task_id] >= task.effort:
                task.completed = True
                self._tasks_completed += 1
                if self._current_step <= task.deadline:
                    reward = 0.90
                    self._on_time.add(task.task_id)
                else:
                    reward = 0.25
            else:
                ratio = self._progress[task.task_id] / task.effort
                reward = _clip(0.1 + 0.3 * ratio)
        else:
            reward = 0.05

        done = (
            self._current_step >= MAX_STEPS
            or all(t.completed for t in self._tasks)
        )

        return TaskSchedulerObservation(
            done=done,
            reward=_clip(reward),
            current_step=self._current_step,
            tasks=[t.to_dict() for t in self._tasks],
            message="ok",
            score=self.grader(),
        )

    @property
    def state(self):
        return self._state