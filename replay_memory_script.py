import sys
import os
import glob

sys.path.append("./language_soft_actor_critic")
sys.path.append("/content/drive/MyDrive/calvinoffon/calvin_env")

from language_soft_actor_critic.replay_memory import ReplayMemory,ExtendedReplayMemory

memory = ExtendedReplayMemory(capacity=1000000, seed=123456)

# Load the latest buffer
files = glob.glob("checkpoints/sac_buffer_calvin*")
latest_file = max(files, key=os.path.getctime)

memory.load_buffer(latest_file)

memory.print_task_counts()


done_items = memory.get_done_items()
print("done_items:\n", len(done_items))
print("\nSample item")
print("state:\n", done_items[0][0])
print("action:\n", done_items[0][1])
print("reward:\n", done_items[0][2])
print("next_state:\n", done_items[0][3])
print("done:\n", done_items[0][4])
print("task_id:\n", done_items[0][5])