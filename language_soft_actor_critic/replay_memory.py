import random
import numpy as np

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

# ReplayMemory 클래스를 상속받는 새로운 클래스 정의
class ExtendedReplayMemory(ReplayMemory):
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

    def print_task_id_counts(self):
        # task_id만 추출
        task_ids = [entry[-1] for entry in self.buffer if entry is not None]
        # Counter를 사용하여 각 task_id의 발생 횟수 계산
        task_id_counts = Counter(task_ids)

        # 결과 출력
        for task_id, count in task_id_counts.items():
            print(f"Task ID {task_id}: {count}개")