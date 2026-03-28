from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import TaskSchedulerAction, TaskSchedulerObservation
except ImportError:
    from models import TaskSchedulerAction, TaskSchedulerObservation


class TaskSchedulerEnv(
    EnvClient[TaskSchedulerAction, TaskSchedulerObservation, State]
):
    """
    Client for the Task Scheduler Environment.

    Example:
        >>> with TaskSchedulerEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     result = env.step(TaskSchedulerAction(task_id=0))
        ...     print(result.observation.message)
    """

    def _step_payload(self, action: TaskSchedulerAction) -> Dict:
        """Convert TaskSchedulerAction to JSON payload."""
        return {
            "task_id": action.task_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TaskSchedulerObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)

        observation = TaskSchedulerObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            current_step=obs_data.get("current_step", 0),
            tasks=obs_data.get("tasks", []),
            message=obs_data.get("message", ""),
            score=obs_data.get("score", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )