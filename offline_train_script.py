import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import sys
import os

sys.path.append("./language_soft_actor_critic")
sys.path.append("/content/drive/MyDrive/calvinoffon/calvin_env")

from language_soft_actor_critic import off2on_sac
from torch.utils.tensorboard import SummaryWriter
from language_soft_actor_critic.replay_memory import ReplayMemory,ExtendedReplayMemory,GoalStateMemory

parser = argparse.ArgumentParser(description='PyTorch Language Condtioned Offline Soft Actor-Critic Args')
parser.add_argument('--env-name', default="calvin",
                    help='Default Calvin')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.00001, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--n_tasks', type=int, default=34, metavar='N',
                    help='encode size (default: 32)')
parser.add_argument('--reduction_dim', type=int, default=200, metavar='N',
                    help='reduction_dim')
parser.add_argument('--out_dim', type=int, default=64, metavar='N',
                    help='reduction_dim')

parser.add_argument('--temp', type=float, default=0.01, metavar='N',
                    help='temp')
            
parser.add_argument('--multiplier', type=float, default=1.1, metavar='N',
                    help='multiplier')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='200')
parser.add_argument('--print_interval', type=int, default=10, metavar='N',
                    help='200')
parser.add_argument('--warm_up_epochs', type=int, default=5, metavar='N',
                    help='10')

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--success_reward', type=float, default=10., metavar='G',
                    help='success_reward default 10.')
parser.add_argument('--cuda', action="store_true",help='run on CUDA (default: False)')

##Dataset config
parser.add_argument('--data_path', type=str, default='./data/training.npz')

args = parser.parse_args()

from dataset.calvin_dataset import CALVIN_dataset
from torch.utils.data import DataLoader
training_dataset = CALVIN_dataset(args.data_path,args.multiplier,args.temp,reduction_dim = args.reduction_dim)
train_data_loader = DataLoader(dataset=training_dataset, batch_size=args.batch_size,shuffle=True, drop_last=True)

##Environment init
import hydra
from hydra import initialize, compose

with initialize(config_path="./calvin_env/conf/"):
  cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper"])
  cfg.env["use_egl"] = False
  cfg.env["show_gui"] = False
  cfg.env["use_vr"] = False
  cfg.env["use_scene_info"] = True
  print(cfg.env)


import custom_calvin_env as calvin #import Custom_Calvin_Env

new_env_cfg = {**cfg.env}
new_env_cfg["tasks"] = cfg.tasks
new_env_cfg.pop('_target_', None)
new_env_cfg.pop('_recursive_', None)
env = calvin.Custom_Calvin_Env(sparse_reward_val=args.success_reward,**new_env_cfg)

#Agent
obs_dim = train_data_loader.dataset.data[0]['robot_obs'].shape[-1]+train_data_loader.dataset.data[0]['scene_obs'].shape[-1]
lang_dim = train_data_loader.dataset.data[0]['emb'].shape[-1]

reduced_lang_embeddings = torch.tensor(training_dataset.reduced_lang_emb)

agent =off2on_sac.Off2On_SAC(obs_dim,args.out_dim,args.reduction_dim,reduced_lang_embeddings,env.action_space,args)

#Tensorboard
writer = SummaryWriter('logs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ExtendedReplayMemory(args.replay_size, args.seed)
goal_memory = GoalStateMemory(args.replay_size, args.seed)

# Training With Offline Data
total_numsteps = 0
updates = 0
device = "cuda" if args.cuda else "cpu"
epochs = args.epochs
#agent.train()


print('\n\n\n\n\n___________________________Begin Training__________________________')

for epoch in range(epochs):
  for episode,pad,task_id in train_data_loader:
    batch_robot_obs = episode['robot_obs'].float().to(device)
    batch_scene_obs = episode['scene_obs'].float().to(device)
    batch_actions = episode['rel_actions'].float().to(device)
    task_ids = task_id.to(device)
    #batch_embeddings = episode['emb'].to(device)
    pad = pad.to(device)
    # batch_r_goals= episode['robot_obs_g'].float().to(device)
    # batch_s_goals= episode['scene_obs_g'].float().to(device)
    batch_r_goals = episode['robot_obs'][:,-1,:].reshape(-1,15).to(device)
    batch_s_goals = episode['scene_obs'][:,-1,:].reshape(-1,24).to(device)


    ##Construct Reward
    reward_bool = ~pad*torch.tensor(args.success_reward)*2.5#).double()) ##[B,L]
    idx_1 = torch.arange(args.batch_size).reshape(args.batch_size,1)
    idx_2 = -torch.sum(~pad, dim=1,keepdim=True)
    
    reward_bool[idx_1, idx_2-1] = 1.5
    reward_bool[idx_1, idx_2-2] = 1.25
    reward_bool[idx_1, idx_2-3] = 1.15
    
    rewards = reward_bool[:,1:]-reward_bool[:,:-1]##[B,L-1]
    batch_obs = torch.cat([batch_robot_obs,batch_scene_obs],dim=-1)
    next_batch_obs = torch.cat([batch_robot_obs[:, 1:, :], batch_scene_obs[:, 1:, :]], dim=-1)
    batch_goals = torch.cat([batch_r_goals,batch_s_goals],dim=-1)

    ### 수정됨, offline에서는 벡터 연산 할 수 있게 수정 필요
    # batch 단위가 아니라 각 timestep마다 replay buffer에 저장
    for b in range(batch_robot_obs.size(0)):  # 배치 사이즈만큼 반복
        for t in range(rewards.size(1)):  # L-1 timestep만큼 반복
            obs = batch_obs[b, t, :].cpu().numpy()
            next_obs = next_batch_obs[b, t, :].cpu().numpy() if t+1 < next_batch_obs.size(1) else None
            action = batch_actions[b, t, :].cpu().numpy()

            reward = rewards[b, t].cpu().item()
            if(pad[b, t+1] == False):
               done = True
            else:
               done = False
            done = pad[b, t+1].cpu().item() if t+1 < pad.size(1) else True

            # Replay buffer에 저장
            task_id_scalar = task_id[b].item()  # b는 배치 내의 인덱스
            training_dataset.get_task_name(task_id_scalar)
            # print(task_id_scalar)
            memory.push_with_task_id(obs, action, reward, next_obs, done, task_id_scalar)

            if(done):
              goal_memory.push(obs, task_id_scalar)
              done = False
              break

    if(epoch < args.warm_up_epochs):
      critic_1_loss, critic_2_loss, policy_loss, ent_loss, latent_loss, alpha = agent.offline_update(batch_obs,batch_actions,task_ids,rewards,pad,batch_goals,updates,True)
    else:
      critic_1_loss, critic_2_loss, policy_loss, ent_loss, latent_loss, alpha = agent.offline_update(batch_obs,batch_actions,task_ids,rewards,pad,batch_goals,updates,False)

    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
    writer.add_scalar('loss/policy', policy_loss, updates)
    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
    writer.add_scalar('loss/latent_loss', latent_loss, updates)
    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
    updates += 1

    if(updates % 10 == 0):
      print(f"Epoch{epoch} - critic_1_loss : {critic_1_loss}, critic_2_loss : {critic_2_loss}, policy_loss : {policy_loss}, ent_loss : {ent_loss},latent_loss : {latent_loss}alpha : {alpha}")
  agent.policy_scheduler.step()
  agent.critic_scheduler.step()
  agent.save_checkpoint(args.env_name)

from datetime import datetime
memory.save_buffer(args.env_name, suffix='offline_train_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
goal_memory.save_buffer(args.env_name, suffix='goal_memory{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))