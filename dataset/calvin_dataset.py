from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import csv


class CALVIN_dataset(Dataset):
    def __init__(self, data_path, multiplier=1.4,temp=0.5,max_len = 64, min_len = 4,pad = True,
    reduction_dim = 200,
    lang_emb_path='/content/drive/MyDrive/calvinoffon/calvin_env/dataset/calvin_debug_dataset/validation/lang_annotations/embeddings.npy'):
        self.data = np.load(data_path, allow_pickle=True)['arr_0']
        self.task = []
        self.max_len = max_len
        self.min_len = min_len
        self.pad = pad
        self.reduction_dim = reduction_dim
        self.data_keys = ['actions', 'rel_actions', 'robot_obs', 'scene_obs'] #'rgb_static']
        self.lang_emb = None
        self.reduced_lang_emb = None

        for data in self.data:
          self.task.append(data.pop('task', None))
        
        ### Language Reduction
        lang_embeddings = np.load(lang_emb_path, allow_pickle=True).item()
        self.task_list = list(lang_embeddings.keys())
        self.task_to_id = {task: id for id, task in enumerate(self.task_list)}


        self.language_reduction(lang_path=lang_emb_path)
        self.make_goals()

        ### Make Prior
        self.multiplier = multiplier
        self.temp = temp
        self.make_task_prob_prior()

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

    def make_goals(self):
        for trajectory in self.data:
          trajectory['robot_obs_g'] = trajectory['robot_obs'][-1,:]
          trajectory['scene_obs_g'] = trajectory['scene_obs'][-1,:]


    def make_task_prob_prior(self):
        for idx,data in enumerate(self.data):
          query_lang = data['emb']#(1,384)
          query_lang /= np.linalg.norm(query_lang,keepdims=True)
          task_id = self.get_task_id(self.task[idx])
          time_step = np.arange(self.max_len) #(L,)
          multiplier_t = self.multiplier**time_step
          ##key (34,lang_dim)
          key = self.lang_emb/np.linalg.norm(self.lang_emb,axis=1,keepdims=True)
          lang_sim = np.einsum('ij,kj->ik', query_lang, self.lang_emb).squeeze() / self.temp # dim (34,)
          lang_sim -= np.max(lang_sim)
          prior = np.einsum('i,j->ij', multiplier_t, lang_sim) # dim (L, 34)
          #log_softmax
          prior = prior-np.max(prior,axis=1,keepdims=True)
          prior = np.exp(prior)
          softmax = prior/np.sum(prior,axis=1,keepdims=True)
          data['prior'] = softmax

    def language_reduction(self,
    lang_path = "/content/drive/MyDrive/calvinoffon/dataset/calvin_debug_dataset/validation/lang_annotations/embeddings.npy",
    txt_path = "/content/drive/MyDrive/calvinoffon/language_soft_actor_critic/Similar_task.csv"):
        lang_embeddings = np.load(lang_path, allow_pickle= True).item()
        lang_instruction = [i for i in lang_embeddings.keys()]
        embeddings_org = [lang_embeddings[i]['emb'].squeeze() for i in lang_instruction]
        sentences = []
        with open(txt_path, 'r') as file:
            file_contents = csv.reader(file)
            for i in file_contents:
                for j in i:
                    sentences.append(j)

        self.lang_emb = np.array([lang_embeddings[i]['emb'].squeeze() for i in lang_instruction])

        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        embeddings_gen = [model.encode(i) for i in sentences]
        X_train_raw = np.asarray(embeddings_org + embeddings_gen)

        #if self.reduction_dim < X_train_raw.shape[1]/2:
        #    print("reduced dim should be larger than #datasets/2. It may hurt representation performances")
        pca = PCA(n_components= 384)
        X_train = X_train_raw - np.mean(X_train_raw)
        X_fit = pca.fit_transform(X_train)
        U1 = pca.components_

        z = []
        for i, x in enumerate(X_train):
            for u in U1[0:7]:
                x = x - np.dot(u.transpose(), x) * u
            z.append(x)
        z = np.asarray(z)

        pca = PCA(n_components = self.reduction_dim)
        X_train = z - np.mean(z)
        X_new_final = pca.fit_transform(X_train)

        pca = PCA(n_components = self.reduction_dim)
        X_new = X_new_final - np.mean(X_new_final)
        X_new = pca.fit_transform(X_new)
        Ufit = pca.components_
        X_new_final = X_new_final - np.mean(X_new_final)

        self.reduced_lang_emb = X_new_final[:34]

  

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

  
