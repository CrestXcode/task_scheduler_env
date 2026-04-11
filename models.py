from typing import List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class Task:
    def __init__(self, task_id, name, priority, effort, deadline, completed=False):
        self.task_id   = task_id
        self.name      = name
        self.priority  = priority
        self.effort    = effort
        self.deadline  = deadline
        self.completed = completed

    def to_dict(self, progress: dict = None) -> dict:
        work_done = (progress or {}).get(self.task_id, 0)
        return {
            "task_id":       self.task_id,
            "name":          self.name,
            "priority":      self.priority,
            "effort":        self.effort,
            "deadline":      self.deadline,
            "completed":     self.completed,
            "work_progress": work_done,   # ← actual progress, not always 0
        }


class TaskSchedulerAction(Action):
    task_id: int = Field(...)


class TaskSchedulerObservation(Observation):
    done:         bool          = False
    reward:       Optional[float] = 0.1   # ← default 0.1 not 0.0
    current_step: int           = 0
    tasks:        List[dict]    = []
    message:      str           = ""
    score:        float         = 0.05    # ← default 0.05 not 0.0