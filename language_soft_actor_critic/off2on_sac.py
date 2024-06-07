#####Offline Actor Critic Definition
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam,SGD
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from language_encoder import GoalConditioned_StateEncoder
import numpy as np
from torch.nn.utils import clip_grad_norm_
from utils import CosineAnnealingWarmUpRestarts

class Off2On_SAC(object):
    def __init__(self, obs_dim,out_dim,reduced_lang_dim,reduced_lang_embdeddings,action_space,args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.sg_encoder =GoalConditioned_StateEncoder(obs_dim,out_dim,tau=self.tau,identity=True).to(self.device) # If identity=True -> encoder = nn.Identity / obs_dim = scene_obs + robot_obs
        # self.sg_enc_optim = Adam(self.sg_encoder.parameters(), lr=0.0001)

        #enc_obs_dim = self.sg_encoder.output_dim + obs_dim
        enc_obs_dim = obs_dim

        self.critic = QNetwork(enc_obs_dim, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=0.0001)

        self.critic_target = QNetwork(enc_obs_dim, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=0.0001)

            self.policy = GaussianPolicy(enc_obs_dim, action_space.shape[0],args.hidden_size,reduced_lang_dim,reduced_lang_embdeddings,action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=0.00001)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(enc_obs_dim, action_space.shape[0],args.hidden_size,reduced_lang_dim,reduced_lang_embdeddings, action_space).to(self.device) #,out_dim
            self.policy_optim = Adam(self.policy.parameters(), lr=0.00001)

        self.policy_scheduler = CosineAnnealingWarmUpRestarts(self.policy_optim, T_0=8, T_mult=1, eta_max=0.00005, T_up=2, gamma=0.5)
        self.critic_scheduler = CosineAnnealingWarmUpRestarts(self.critic_optim, T_0=8, T_mult=1, eta_max=0.00005, T_up=2, gamma=0.5)

    def select_action(self, enc_state, curr_goals, evaluate=False):
        enc_state = enc_state.to(self.device)
        #print(sstate.dtype)
        task_id = task_id.to(self.device)

        if evaluate is False:
            action, _, _ = self.policy.sample(enc_state, curr_goals)
        else:
            _, _, action = self.policy.sample(enc_state, curr_goals)
        return action.detach().cpu().numpy()[0]

    def offline_update(self,observations,actions,task_ids,reward,mask,goals,updates,warm_up=True):
      #observations.shape = (B,L,obs_dim)
      #actions.shape = (B,L,action_shape)
      #task_ids = (B,1)
      B,L_o,_ = observations.shape
      latent_obs = observations.reshape(B*L_o,-1) #(B*L_o,state_dim)
      latent_goals = goals.reshape(B,-1)
      latent_masks = mask.reshape(B*L_o,-1)

      ##Training Phase 1-1 : Representation Learning - Latent
      #latent_lang = lang.repeat(1,L_o,1).reshape(B*L_o,-1)#(B*L_o,lang_dim)
    #   latent_loss = self.sg_encoder.loss(latent_obs,latent_goals,L_o,self.gamma,latent_masks)

    #   self.sg_enc_optim.zero_grad()
    #   latent_loss.backward()#retain_graph=True)
    #   self.sg_enc_optim.step()

      if(warm_up==False):
        B,L = reward.shape #여기서 L 은 L-1
        curr_obs = observations[:,:-1,:].reshape(B*L,-1) #(B*L,state_dim)
        curr_action = actions[:,:-1,:].reshape(B*L,-1) #(B*L,action_dim)
        curr_goals = goals.reshape(B,-1)  #(B, 39)
        curr_task_id = task_ids.repeat(1,L).flatten()

        next_obs = observations[:,1:,:].reshape(B*L,-1) #(B*L,state_dim)
        next_action = actions[:,1:,:].reshape(B*L,-1) #(B*L,action_dim)
        next_goals = goals.reshape(B,-1)
        next_task_id = task_ids.repeat(1,L).flatten()

        # curr_actor_in,curr_goal_in,_= self.sg_encoder(curr_obs,curr_goals) #(B*L,enc_dim)
        # next_actor_in,next_goal_in,_ = self.sg_encoder(next_obs,next_goals)#(B*L,enc_dim)
        # curr_actor_in = curr_actor_in.detach()
        # next_actor_in = next_actor_in.detach()

        # curr_actor_in = torch.cat([curr_obs, curr_actor_in], dim=-1).detach()   #(B*L + B, 39)
        # next_actor_in = torch.cat([next_obs, next_actor_in], dim=-1).detach()

        with torch.no_grad():
          next_state_action, next_state_log_pi, _ = self.policy.sample(
              next_obs, curr_goals)
          qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_state_action)
          min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
          next_q_value = reward.reshape(B*L,-1) + mask[:,:-1].reshape(B*L,-1) * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(curr_obs, curr_action)# Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()#retain_graph=True)
        self.critic_optim.step()
        # clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        pi, log_pi, _ = self.policy.sample(curr_obs, curr_goals)
        qf1_pi, qf2_pi = self.critic(curr_obs,pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]


        self.policy_optim.zero_grad()
        policy_loss.backward()#retain_graph=True)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            clip_grad_norm_([self.log_alpha], max_norm=1.0)
            self.alpha_optim.step()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

    
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(),0, alpha_tlogs.item()

      return 0.,0.,0.,0.,latent_loss.item(),0.
   
    def encode_latent(self, state, task_id,latent_prior):
        state = torch.FloatTensor(np.concatenate([state['robot_obs'],state['scene_obs']])).unsqueeze(0).to(device = self.device)
        #state2 = torch.FloatTensor(state['scene_obs']).to(device = self.device)
        # print(lang_emb.shape)
        #state = torch.cat((state1, state2)).unsqueeze(0)
        rep_state,_,_,_ = self.state_lang_encoder(state,torch.tensor(task_id),latent_prior.to(device = self.device))
        return rep_state

    def online_update(self, lang, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch_lang = [self.encode_latent(state, lang) for state in state_batch]
        next_state_batch_lang = [self.encode_latent(state, lang) for state in next_state_batch]

        state_batch_lang = [self.state_concat(state, lang) for state in state_batch]
        next_state_batch_lang = [self.state_concat(state, lang) for state in next_state_batch]

        state_batch = torch.stack(state_batch_lang, dim=0).to(self.device)
        next_state_batch = torch.stack(next_state_batch_lang, dim=0).to(self.device)

        # state_batch = torch.FloatTensor(state_batch_lang).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch_lang).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss


        self.critic_optim.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        self.state_lang_optim.zero_grad()
        self.state_lang_optim.backward()
        self.state_lang_optim.step()


        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'encoder_state_dict': self.sg_encoder.state_dict(),
                    'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),

                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.sg_encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

        print()
