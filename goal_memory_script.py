import sys
import os
import glob
from collections import Counter
import numpy as np

sys.path.append("./language_soft_actor_critic")
sys.path.append("/content/drive/MyDrive/calvinoffon/calvin_env")

from language_soft_actor_critic.replay_memory import GoalStateMemory

memory = GoalStateMemory(capacity=1000000, seed=123456)

# Load the latest buffer
files = glob.glob("checkpoints/sac_buffer_calvin_goal_memory*")
latest_file = max(files, key=os.path.getctime)

memory.load_buffer(latest_file)

task_ids = [entry[-1] for entry in memory.buffer if entry is not None]
# Counter를 사용하여 각 task_id의 발생 횟수 계산
task_id_counts = Counter(task_ids)

for task_id, count in task_id_counts.items():
    print(f"Task {task_id}: {count} occurrences")

task_id = 10
batch_size = 10

batch = memory.sample(task_id, batch_size)

for state in batch:
    print(f"Task {task_id}: {state}")