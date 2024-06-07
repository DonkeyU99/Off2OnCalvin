import os
import random
import numpy as np
import pickle

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity

from collections import Counter

class GoalStateMemory(ReplayMemory):
    def push(self, state, task_id):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, task_id)
        self.position = (self.position + 1) % self.capacity

    def sample(self, task_id, batch_size = 1):
        goal_state = [entry[0] for entry in self.buffer if entry is not None and entry[1] == task_id]
        if len(goal_state) < batch_size:
            raise ValueError("Not enough goal states for the task")

        batch = random.sample(goal_state, batch_size)
        return batch

# Goal상태의 state를 저장
class ExtendedReplayMemory(ReplayMemory):
    def __init__(self, capacity, seed):
        super(ExtendedReplayMemory, self).__init__(capacity, seed)
        self.set_task_dict()
        
    def get_task_name(self, task_id):
        return self.task_dict.get(task_id, "Invalid task id")
    
    def push_with_task_id(self, state, action, reward, next_state, done, task_id):
        # 상속받은 push 함수를 사용하여 구현
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, task_id)
        self.position = (self.position + 1) % self.capacity

    def sample_with_task_id(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, task_id = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def get_sorted_task_counts(self):
        # task_id만 추출
        task_ids = [entry[-1] for entry in self.buffer if entry is not None]
        # Counter를 사용하여 각 task_id의 발생 횟수 계산
        task_id_counts = Counter(task_ids)
        # count에 대한 오름차순으로 정렬
        sorted_task_counts = task_id_counts.most_common()[::-1]
        return sorted_task_counts
    
    def print_task_counts(self):
        for task_id, count in self.get_sorted_task_counts():
            print(f"[{task_id}] {self.get_task_name(task_id)} : {count}개")

    def get_done_items(self):
        done_items = [
            item for item in self.buffer if item is not None and item[4]]
        return done_items

    def set_task_dict(self):
        self.task_dict = {
            0: "rotate_red_block_right",
            1: "rotate_red_block_left",
            2: "rotate_blue_block_right",
            3: "rotate_blue_block_left",
            4: "rotate_pink_block_right",
            5: "rotate_pink_block_left",
            6: "push_red_block_right",
            7: "push_red_block_left",
            8: "push_blue_block_right",
            9: "push_blue_block_left",
            10: "push_pink_block_right",
            11: "push_pink_block_left",
            12: "move_slider_left",
            13: "move_slider_right",
            14: "open_drawer",
            15: "close_drawer",
            16: "lift_red_block_table",
            17: "lift_blue_block_table",
            18: "lift_pink_block_table",
            19: "lift_red_block_slider",
            20: "lift_blue_block_slider",
            21: "lift_pink_block_slider",
            22: "lift_red_block_drawer",
            23: "lift_blue_block_drawer",
            24: "lift_pink_block_drawer",
            25: "place_in_slider",
            26: "place_in_drawer",
            27: "push_into_drawer",
            28: "stack_block",
            29: "unstack_block",
            30: "turn_on_lightbulb",
            31: "turn_off_lightbulb",
            32: "turn_on_led",
            33: "turn_off_led"
        }