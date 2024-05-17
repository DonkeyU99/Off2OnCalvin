import numpy as np
from tqdm import tqdm

def dataset_rezip(data_path,save_path='./data',file_name='training',data_keys=None):
    data = []
    if data_keys is None:
        data_keys = ['actions', 'rel_actions', 'robot_obs', 'scene_obs'] #'rgb_static']

    lang_annotation_dicts = np.load(data_path + '/lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()

    trajectory_lang_data = []

    for i, task in enumerate(lang_annotation_dicts['language']['task']):
        embedding = lang_annotation_dicts['language']['emb'][i,:,:]
        traj_idx = lang_annotation_dicts['info']['indx'][i]
        trajectory_lang_data.append({'task' : task,
                                     'emb' : embedding,
                                     'traj_idx' : traj_idx})

    for traj in tqdm(trajectory_lang_data):
        trajectory = dict()
        for key in data_keys:
            trajectory[key] = []
        for j in range(traj['traj_idx'][0], traj['traj_idx'][1]):
            episode = np.load(data_path + f'/episode_0{j}.npz')
            for key in data_keys:
                trajectory[key].append(episode[key])

        trajectory['emb'] = traj['emb']
        trajectory['task'] = traj['task']

        for key in trajectory.keys():
            trajectory[key] = np.array(trajectory[key])

        data.append(trajectory)
    
    np.savez(save_path + '/' + file_name, data)