import gym
from gym import spaces
import numpy as np
import hydra

import sys
from calvin_env.envs.play_table_env import PlayTableSimEnv

class Custom_Calvin_Env(PlayTableSimEnv):
    def __init__(self,
                 tasks: dict = {},
                 use_rel_action = True,
                 image_based = False,
                 sparse_reward_val = 10.,
                 **kwargs):
        super(Custom_Calvin_Env, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Dict({'robot_obs': spaces.Box(low=-1, high=1, shape=(15,)),
                                                  'scene_obs': spaces.Box(low=-1, high=1, shape=(24,))})
        # action_space, observation_space define
        '''if use_rel_action:
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        else:
            self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        if image_based:
            self.observation_space = spaces.Box(low=0, high=255, shape=(200,200,3))
        else:
            self.observation_space = spaces.Dict({'robot_obs': spaces.Box(low=-1, high=1, shape=(15,)),
                                                  'scene_obs': spaces.Box(low=-1, high=1, shape=(24,))})

        #self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))
        '''
        # 가능한 tasks 전부 -> 34개
        self.tasks = hydra.utils.instantiate(tasks)
        '''
        self.tasks.tasks : 34개 task 모여있는 dict (task : 어떤 동작(?))
        self.tasks.task_to_id : (task : id) dict
        self.tasks.id_to_task : (id : task) dict

        self.tasks.get_task_info(start_info, end_info) : start_info와 end_info 주면 achieved task를 알려줌
        self.tasks.get_task_for_set : 주어진 task에 대해 check
        '''
        self.current_task = [] # 현재의 task
        self.sparse_reward_val = sparse_reward_val

    def sample_task(self):
        id = np.random.randint(self.tasks.num_tasks)
        self.current_task = [self.tasks.id_to_task[id]]

    '''
    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        return obs

    '''
    def reset(self, robot_obs=None, scene_obs=None):
        self.scene.reset(scene_obs)
        self.robot.reset(robot_obs)
        self.start_info = self.get_info()
        self.sample_task()
        self.p.stepSimulation(physicsClientId=self.cid)
        return self.get_obs()

    '''
    def get_obs(self):
        if self.image_based:
            rgb_obs, _ = self.get_camera_obs()
            return rgb_obs['rgb_static']
        else:
            robot_obs, robot_info = self.robot.get_observation()
            scene_obs = self.scene.get_obs()
            obs = {"robot_obs": robot_obs, "scene_obs": scene_obs}
            return obs
    '''
    def _success(self):
        """ Returns a boolean indicating if the task was performed correctly """
        current_info = self.get_info()
        #task_filter = ["move_slider_left"]
        task_filter = self.current_task
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return task_filter[0] in task_info

    def _reward(self):
        """ Returns the reward function that will be used
        for the RL algorithm """
        reward = int(self._success()) * self.sparse_reward_val
        r_info = {'reward': reward}
        return reward, r_info

    def _termination(self):
        """ Indicates if the robot has reached a terminal state """
        success = self._success()
        done = success
        d_info = {'success': success}
        return done, d_info

    def step(self, action):
        self.robot.apply_action(action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)

        return obs, reward, done, info