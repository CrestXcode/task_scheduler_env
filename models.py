from typing import List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class Task:
    """Represents a single task in the scheduler."""
    def __init__(self, task_id: int, name: str, priority: str,
                 effort: int, deadline: int, completed: bool = False):
        self.task_id = task_id
        self.name = name
        self.priority = priority
        self.effort = effort
        self.deadline = deadline
        self.completed = completed
        self._deadline_missed = False

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "priority": self.priority,
            "effort": self.effort,
            "deadline": self.deadline,
            "completed": self.completed
        }


class TaskSchedulerAction(Action):
    """Action — agent picks which task ID to work on this step."""
    task_id: int = Field(..., description="ID of the task to work on this step")


class TaskSchedulerObservation(Observation):
    """Observation returned after every step."""
    done: bool = Field(default=False, description="Is the episode over?")
    reward: Optional[float] = Field(default=None, description="Reward this step")
    current_step: int = Field(default=0, description="Current time step")
    tasks: List[dict] = Field(default_factory=list, description="All tasks and their status")
    message: str = Field(default="", description="Feedback message to agent")
    score: float = Field(default=0.0, description="Running score 0.0-1.0")