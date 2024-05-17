import numpy as np
from torch.utils.data import Dataset

class CALVIN_dataset(Dataset):
    def __init__(self, data_path, max_len = 64, min_len = 4, pad = True,
    lang_emb_path='/content/drive/MyDrive/calvinoffon/calvin_env/dataset/calvin_debug_dataset/validation/lang_annotations/embeddings.npy'):
        self.data = np.load(data_path, allow_pickle=True)['arr_0']
        self.task = []
        self.max_len = max_len
        self.min_len = min_len
        self.pad = pad
        self.data_keys = ['actions', 'rel_actions', 'robot_obs', 'scene_obs'] #'rgb_static']

        for data in self.data:
          self.task.append(data.pop('task', None))
        
        ### 수정됨
        lang_embeddings = np.load(lang_emb_path, allow_pickle=True).item()
        self.task_list = list(lang_embeddings.keys())
        self.task_to_id = {task: id for id, task in enumerate(self.task_list)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ### 수정됨
        # trajectory.pop()을 수행한 다음에도 task_id를 구할 수 있도록 .copy 수행
        trajectory = self.data[idx]
        if len(trajectory['actions']) < self.max_len and self.pad:
            trajectory,pad_vector = self.padding(trajectory)
        else:
          pad_vector = np.ones(self.max_len,dtype=bool)
          pad_vector[-1] = False

        task_id = self.get_task_id(self.task[idx])

        return trajectory,pad_vector,task_id
    ### 추가됨
    def get_task_id(self, task):
        task_str = str(task)
        return self.task_to_id.get(task_str, -1)
    ### 추가됨
    def get_task_name(self, task_id):
        for task, id in self.task_to_id.items():
            if id == task_id:
                return task
        raise Exception("Invalid task_id")

    def padding(self, trajectory):
        traj_len = len(trajectory['actions'])
        pad_len = self.max_len - traj_len
        for key in self.data_keys:
            shape = list(trajectory[key].shape)
            shape[0] = pad_len
            pad = np.zeros(tuple(shape),dtype=np.float64)
            trajectory[key] = np.concatenate((trajectory[key], pad), axis=0)
        pad_vector = np.zeros(self.max_len,dtype=bool)
        pad_vector[:traj_len] = True
        return trajectory,pad_vector

if __name__ == "__main__":
  from torch.utils.data import DataLoader
  training_dataset = CALVIN_dataset('./data/training.npz')
  train_data_loader = DataLoader(dataset=training_dataset, batch_size=2,shuffle=True)
  for i,pad_vector,task_id in train_data_loader:
    
    bool_rew = (~pad_vector)*10
    print("REWARD")
    print(bool_rew[:,1:]-bool_rew[:,:-1])
    print("TASK ID")
    print(task_id)
    break
