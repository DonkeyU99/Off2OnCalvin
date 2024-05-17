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
from language_soft_actor_critic.replay_memory import ReplayMemory,ExtendedReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Language Condtioned Offline Soft Actor-Critic Args')
parser.add_argument('--env-name', default="calvin",
                    help='Default Calvin')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--enc_dim', type=int, default=32, metavar='N',
                    help='encode size (default: 32)')
parser.add_argument('--q_hidden_dim', type=int, default=32, metavar='N',
                    help='q_hidden_dim (default: 64)')

parser.add_argument('--max_epi_length', type=int, default=100, metavar='N')

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--success_reward', type=float, default=10., metavar='G',
                    help='success_reward default 10.')
parser.add_argument('--cuda', action="store_true",help='run on CUDA (default: False)')

parser.add_argument('--start_steps', type=int, default=10000, metavar='N')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N')           

##Dataset config
parser.add_argument('--lang_emb_path', type=str, default='/content/drive/MyDrive/calvinoffon/calvin_env/dataset/calvin_debug_dataset/validation/lang_annotations/embeddings.npy')

args = parser.parse_args()

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

import custom_calvin_env as calvin#import Custom_Calvin_Env

new_env_cfg = {**cfg.env}
new_env_cfg["tasks"] = cfg.tasks
new_env_cfg.pop('_target_', None)
new_env_cfg.pop('_recursive_', None)
env = calvin.Custom_Calvin_Env(sparse_reward_val=args.success_reward,**new_env_cfg)

env.reset()
task_to_vec = np.load(args.lang_emb_path, allow_pickle=True).item()

#Agent
obs_dim = env.observation_space["robot_obs"].shape[-1] + env.observation_space["scene_obs"].shape[-1]
lang_dim = task_to_vec[env.current_task[0]]['emb'].shape[-1]
enc_dim = args.enc_dim
q_hidden_dim = args.q_hidden_dim

#Tensorboard
writer = SummaryWriter('logs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

agent =off2on_sac.Off2On_SAC(obs_dim,lang_dim,enc_dim,q_hidden_dim,env.action_space,args)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Online SAC
total_numsteps = 0
updates = 0

print('\n\n\n\n\n___________________________Begin Training__________________________')

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    print(env.current_task[0])
    lang = torch.tensor(task_to_vec[env.current_task[0]]["emb"])

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state, lang)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.online_update(lang, memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
                print("updates;", updates)

        action[-1] = (int(action[-1] >= 0) * 2) - 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        print(total_numsteps, "\t", done)

        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

        if episode_steps > args.max_epi_length:
            done = True

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, lang, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()
