import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openenv.core.env_server.http_server import create_app
from models import TaskSchedulerAction, TaskSchedulerObservation
from server.task_scheduler_environment import HardTaskScheduler

app = create_app(HardTaskScheduler, TaskSchedulerAction, TaskSchedulerObservation, env_name="task_scheduler", max_concurrent_envs=1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)