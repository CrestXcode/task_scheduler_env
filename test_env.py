from client import TaskSchedulerEnv, TaskSchedulerAction

with TaskSchedulerEnv(base_url="http://localhost:8000").sync() as env:
    # Reset
    result = env.reset()
    print("After reset:")
    print(f"  Message: {result.observation.message}")
    print(f"  Tasks: {len(result.observation.tasks)} tasks")
    print()

    # Step through all easy tasks
    for task_id in range(5):
        result = env.step(TaskSchedulerAction(task_id=task_id))
        print(f"Step {task_id + 1}:")
        print(f"  Message: {result.observation.message}")
        print(f"  Reward: {result.reward}")
        print(f"  Score: {result.observation.score}")
        print(f"  Done: {result.done}")
        print()
        if result.done:
            break