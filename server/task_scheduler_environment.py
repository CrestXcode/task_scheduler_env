import uuid
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TaskSchedulerAction, TaskSchedulerObservation, Task
except ImportError:
    from models import TaskSchedulerAction, TaskSchedulerObservation, Task


# Global difficulty store — set via /set_difficulty endpoint
_difficulty_store = {"current": "easy"}


TASK_TEMPLATES = {
    "easy": [
        {"name": "Reply to emails",      "priority": "high",   "effort": 1, "deadline": 4},
        {"name": "Update status report", "priority": "medium", "effort": 1, "deadline": 5},
        {"name": "Review pull request",  "priority": "high",   "effort": 1, "deadline": 3},
        {"name": "Schedule meeting",     "priority": "low",    "effort": 1, "deadline": 6},
        {"name": "Fix typo in docs",     "priority": "low",    "effort": 1, "deadline": 7},
        {"name": "Team sync preparation", "priority": "medium", "effort": 1, "deadline": 4},  # 6th task
    ],
    "medium": [
        {"name": "Write unit tests",        "priority": "high",   "effort": 2, "deadline": 5},
        {"name": "Code review sprint",      "priority": "high",   "effort": 2, "deadline": 4},
        {"name": "Update dependencies",     "priority": "medium", "effort": 2, "deadline": 6},
        {"name": "Prepare presentation",    "priority": "high",   "effort": 2, "deadline": 3},
        {"name": "Debug production issue",  "priority": "high",   "effort": 2, "deadline": 2},
        {"name": "Refactor module",         "priority": "medium", "effort": 3, "deadline": 7},
        {"name": "Document API endpoints",  "priority": "low",    "effort": 2, "deadline": 8},
        {"name": "Team sync preparation",   "priority": "medium", "effort": 1, "deadline": 4},
    ],
    "hard": [
        {"name": "Migrate database schema",  "priority": "high",   "effort": 3, "deadline": 4},
        {"name": "Security audit report",    "priority": "high",   "effort": 3, "deadline": 3},
        {"name": "Deploy to production",     "priority": "high",   "effort": 2, "deadline": 2},
        {"name": "Incident post-mortem",     "priority": "high",   "effort": 2, "deadline": 3},
        {"name": "Optimize slow queries",    "priority": "medium", "effort": 3, "deadline": 5},
        {"name": "Implement OAuth flow",     "priority": "high",   "effort": 3, "deadline": 4},
        {"name": "Load testing report",      "priority": "medium", "effort": 2, "deadline": 4},
        {"name": "Onboard new engineer",     "priority": "medium", "effort": 2, "deadline": 5},
        {"name": "Quarterly review prep",    "priority": "high",   "effort": 3, "deadline": 3},
        {"name": "Fix critical bug in prod", "priority": "high",   "effort": 2, "deadline": 1},
    ],
}


class TaskSchedulerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._tasks: list[Task] = []
        self._current_step: int = 0
        self._difficulty: str = "easy"
        self._score: float = 0.0
        self._tasks_completed: int = 0
        self._tasks_failed: int = 0
        self._work_progress: dict[int, int] = {}
        self._urgent_task_injected: bool = False

    def _build_tasks(self, difficulty: str) -> list[Task]:
        templates = TASK_TEMPLATES[difficulty]
        tasks = []
        for i, t in enumerate(templates):
            tasks.append(Task(
                task_id=i,
                name=t["name"],
                priority=t["priority"],
                effort=t["effort"],
                deadline=t["deadline"],
                completed=False,
            ))
        return tasks

    def grader(self) -> float:
        """Return final grade strictly between 0.0 and 1.0 (exclusive)."""
        total = len(self._tasks)
        if total == 0:
            return 0.5
        raw = self._tasks_completed / total
        # Clamp to strictly (0, 1) — never 0.0 or 1.0
        return round(max(0.01, min(0.99, raw)), 2)

    def reset(self, difficulty: str = "easy") -> TaskSchedulerObservation:
        self._difficulty = difficulty
        self._tasks = self._build_tasks(difficulty)
        self._current_step = 0
        self._score = 0.0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._work_progress = {t.task_id: 0 for t in self._tasks}
        self._urgent_task_injected = False
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)

        return TaskSchedulerObservation(
            done=False,
            reward=None,
            current_step=0,
            tasks=[t.to_dict() for t in self._tasks],
            message=f"Episode started! Difficulty: {difficulty}. Complete {len(self._tasks)} tasks before their deadlines.",
            score=0.0,
        )

    def step(self, action: TaskSchedulerAction) -> TaskSchedulerObservation:
        self._current_step += 1
        self._state.step_count += 1
        reward = 0.0
        message = ""

        if (self._difficulty == "hard" and self._current_step == 3 and not self._urgent_task_injected):
            urgent = Task(
                task_id=len(self._tasks),
                name="URGENT: Server is down!",
                priority="high",
                effort=1,
                deadline=self._current_step + 2,
                completed=False,
            )
            self._tasks.append(urgent)
            self._work_progress[urgent.task_id] = 0
            self._urgent_task_injected = True
            message += " URGENT task injected — reprioritise now! "

        task_id = action.task_id
        valid_ids = [t.task_id for t in self._tasks if not t.completed]

        if task_id not in valid_ids:
            reward = -0.2
            message += f"Invalid task_id {task_id}. Choose from {valid_ids}."
        else:
            task = next(t for t in self._tasks if t.task_id == task_id)
            self._work_progress[task_id] += 1
            progress = self._work_progress[task_id]
            steps_left = task.deadline - self._current_step
            reward += 0.1
            if task.priority == "high":
                reward += 0.1
            if progress >= task.effort:
                task.completed = True
                self._tasks_completed += 1
                if self._current_step <= task.deadline:
                    reward += 1.0
                    message += f"'{task.name}' completed on time! +1.0 "
                else:
                    reward += 0.3
                    message += f"'{task.name}' completed late. +0.3 "
            else:
                remaining = task.effort - progress
                message += (f"Working on '{task.name}' ({progress}/{task.effort} steps done, {steps_left} steps until deadline). ")

        for task in self._tasks:
            if (not task.completed and task.deadline < self._current_step and not getattr(task, '_deadline_missed', False)):
                reward -= 0.3
                task.completed = True
                task._deadline_missed = True
                self._tasks_failed += 1
                if task.task_id in self._work_progress:
                    self._work_progress.pop(task.task_id)
                message += f"'{task.name}' deadline missed! -0.3 "

        for task in self._tasks:
            if (not task.completed and task.priority == "high" and task.task_id != task_id and task.deadline - self._current_step <= 2):
                reward -= 0.1
                message += f"'{task.name}' is urgent and being ignored! "

        self._score = self.grader()
        reward = round(reward, 2)
        all_done = all(t.completed for t in self._tasks)
        done = all_done or self._current_step >= 20

        if done:
            if all_done:
                message += f"All tasks done! Final score: {self._score}"
            else:
                message += f"Time's up! Final score: {self._score}"

        return TaskSchedulerObservation(
            done=done,
            reward=reward,
            current_step=self._current_step,
            tasks=[t.to_dict() for t in self._tasks],
            message=message.strip(),
            score=self._score,
        )

    @property
    def state(self) -> State:
        return self._state


class EasyTaskScheduler(TaskSchedulerEnvironment):
    def reset(self, difficulty: str = "easy") -> TaskSchedulerObservation:
        return super().reset(difficulty="easy")


class MediumTaskScheduler(TaskSchedulerEnvironment):
    def reset(self, difficulty: str = "medium") -> TaskSchedulerObservation:
        return super().reset(difficulty="medium")


class HardTaskScheduler(TaskSchedulerEnvironment):
    def reset(self, difficulty: str = "hard") -> TaskSchedulerObservation:
        return super().reset(difficulty="hard")