import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from env import EADEnv
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.data import DataLoader
from logger import Logger

import utils.modelTool as modelTool
from utils.loss import ComputeLoss

class PPO:
    def __init__(self, env:EADEnv, batch_size) -> None:
        self.env = env     
        self.rollout_batch_size = batch_size
        device = torch.device('cuda:0')
        self.sensory = modelTool.get_det_model(pretrain_weights='checkpoints/yolo_carla.pt', freeze = 17, device=device)
        self.sensory.eval()
        modelTool.transfer_paramaters(pretrain_weights='checkpoints/yolo_carla.pt', detModel=self.sensory)

        self.compute_loss = ComputeLoss(self.sensory)

        self.ead = modelTool.get_ead_model(max_steps=4)
        self.ead.load_state_dict(torch.load('checkpoints/ead_offline_small.pt'))
        self.ead.eval()
        print(self.ead.action_scaling)
        
        self.actor = self.ead.action_decoder
        self.critic = self.ead.value_decoder

        self.cov_var = torch.tensor([15.**2,60.**2]).cuda()*0.05
        self.cov_mat = torch.diag(self.cov_var)

        self._init_hyperparameters()

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        
        self.return_logger = Logger('checkpoints/ead_return')
        self.acc_logger = Logger('checkpoints/ead_acc')
        self.eval_acc_cln_logger = Logger('checkpoints/ead_eval_acc_cln')
        self.eval_acc_adv_logger = Logger('checkpoints/ead_eval_acc_adv')
        self.critic_logger = Logger('checkpoints/ead_critic_loss')


    def _init_hyperparameters(self):
        
        self.max_timesteps_per_episode = 3
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.lr = 1e-4
    
    @torch.no_grad()
    def get_actions(self, obs, targets):
        obs = torch.tensor(obs).cuda()
        targets = torch.tensor(targets).cuda()
        refined_feats = self.ead(obs)
        mean = self.ead.get_action(refined_feats)
        _, train_out = self.sensory.ead_stage_2(refined_feats)
        loss, _ = self.compute_loss(train_out, targets[:,-1,:], loss_items=['box', 'obj'])  # loss scaled by batch_size

        dist = MultivariateNormal(mean,self.cov_mat)   
        action = dist.sample()
        action[:,0] = torch.clamp(action[:,0], -15, 15)
        action[:,1] = torch.clamp(action[:,1], -60, 60)
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob.cpu().numpy(), loss
    
    @torch.no_grad()
    def compute_advance(self, batch_obs, batch_rtgs):
        # Compute the estimated average performance of strategy [π] from the critic in the outer iteration (the critic that watchs the online game)
        # Compute the reward to go from the actor in the outer iteration (the actor that interacts with the environment)
        # No gradident flows in the environment
        # Advantage function: measure the improvement of taking action [a] in state [s] over following the average performance of strategy [π].
        refined_feats = self.ead(batch_obs)
        V = self.ead.get_value(refined_feats)
        V = V.squeeze(1) # [B*T]
        A_k = batch_rtgs - V
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        return A_k

    def evaluate(self, batch_obs, batch_acts):
        # Compute the estimated average performance of strategy [π] from the critic in the inner iteration (the critic that learns the game recording)
        # Gradident flows and loss backwards in learning procedure to minimize the difference between the estimate value and the reward to go  
        # Compute the action probability from the actor in the inner iteration (the actor that learns the game recording)
        # Gradident flows and loss backwards in learning procedure to maximize the advantage function (increase the probability of taking a better action)
        # The critic and actor come from the previous outer iteration and go to the next outer iteration (play a game -> learn -> play again)   
        with torch.no_grad():
            refined_feats = self.ead(batch_obs)
        refined_feats = refined_feats.clone().detach().cpu().numpy()
        
        refined_feats = torch.tensor(refined_feats).cuda()
        V = self.ead.get_value(refined_feats)
        V = V.squeeze(1)
        mean = self.ead.get_action(refined_feats)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + self.gamma * discounted_reward
                batch_rtgs.insert(0, discounted_reward)
        return batch_rtgs

    @torch.no_grad()
    def rollout(self, car_idxs):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []

        for car_idx in car_idxs:
            obs, info = self.env.reset(car_idx=car_idx) # init obs, [1, S, F]
            for ep_t in range(self.max_timesteps_per_episode):
                batch_obs.append(obs)
                action, log_prob, loss = self.get_actions(obs[:,:ep_t+1], info['annotations'][:,ep_t:ep_t+1]) # RL model works in this function [1,]
                obs, _, _, _ = self.env.step(action) # next obs, [1, S, F]
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if ep_t == 0:
                    pre_loss = loss.cpu().numpy()
                else:
                    rew = pre_loss - loss.cpu().numpy()
                    batch_rews.append(rew) # calculate reward_t at t+1
                    pre_loss = loss.cpu().numpy()
            _, _, loss = self.get_actions(obs[:,:ep_t+2], info['annotations'][:,ep_t+1:ep_t+2]) # obtain acc and loss from the last obs.
            rew = pre_loss - loss.cpu().numpy()
            batch_rews.append(rew) # calculate reward_t at t+1

        
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float) # [T*B, 1, S, FC, FH, FW]
        S, FC, FH, FW = batch_obs.shape[2],batch_obs.shape[3],batch_obs.shape[4],batch_obs.shape[5]
        batch_obs = batch_obs.unsqueeze(1).reshape(self.max_timesteps_per_episode, self.rollout_batch_size, S, FC, FH, FW) #[T, B, S, FC, FH, FW]
        batch_obs = batch_obs.permute(1, 0, 2, 3, 4, 5) # [B, T, S, FC, FH, FW]
        batch_obs = batch_obs.reshape(-1, S, FC, FH, FW) # [B*T, S, FC, FH, FW]

        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float) # [T*B, 1, A]
        batch_acts = batch_acts.unsqueeze(1).reshape(self.max_timesteps_per_episode, self.rollout_batch_size, batch_acts.shape[2])# [T, B, A]
        batch_acts = batch_acts.permute(1, 0, 2) # [B, T, A]
        batch_acts = batch_acts.reshape(-1, batch_acts.shape[2]) # [B*T, A]
        
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float) # [T*B, 1]
        batch_log_probs = batch_log_probs.unsqueeze(1).reshape(self.max_timesteps_per_episode, self.rollout_batch_size,) # [T, B]
        batch_log_probs = batch_log_probs.permute(1, 0) # [B, T]
        batch_log_probs = batch_log_probs.reshape(-1) # [B*T]
        
        batch_rews = torch.tensor(np.array(batch_rews), dtype=torch.float) # [T*B, 1] 
        batch_rews = batch_rews.unsqueeze(1).reshape(self.max_timesteps_per_episode, self.rollout_batch_size,) # [T, B]       
        batch_rews = batch_rews.permute(1, 0) # [B, T]

        self.return_logger.add_value(batch_rews.sum()/batch_rews.shape[0])
        self.return_logger.plot()

        batch_rtgs = torch.tensor(self.compute_rtgs(batch_rews), dtype=torch.float)

        # print('batch_obs.shape', batch_obs.shape)
        # print('batch_acts.shape', batch_acts.shape)
        # print('batch_log_probs.shape', batch_log_probs.shape)
        # print('batch_rtgs.shape', batch_rtgs.shape)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs

    def learn(self, epoch_number):
        for epoch in range(epoch_number):
            # acc_clean = eval_clean(self.actor_ead)
            # self.eval_acc_cln_logger.add(acc_clean)
            # self.eval_acc_cln_logger.plot()
            # acc_adv = eval_adv(self.actor_ead, 'swin_small_patch4_window7_224', 'MeshAdv', 0.2, 100)
            # self.eval_acc_adv_logger.add(acc_adv)
            # self.eval_acc_adv_logger.plot()

            # self.checkpoints_manager.save_best_checkpoints(epoch, self.lr, self.rollout_batch_size, acc_adv, 'None')
            car_dataset = np.array(np.arange(0,41).tolist()*10)
            random.shuffle(car_dataset)
            car_dataset = car_dataset[0:self.rollout_batch_size*(len(car_dataset)//self.rollout_batch_size)]
            car_dataset = car_dataset.reshape(len(car_dataset)//self.rollout_batch_size, -1)
            car_dataset = car_dataset.tolist()

            for car_idxs in car_dataset:
                print(f'epoch: {epoch}, iteration: {car_idxs}')
                self.ead.eval()
                batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.rollout(car_idxs)
                batch_obs = batch_obs.cuda()
                batch_acts = batch_acts.cuda()
                batch_log_probs = batch_log_probs.cuda()
                batch_rtgs = batch_rtgs.cuda()
                A_k = self.compute_advance(batch_obs, batch_rtgs)
                for _ in range(self.n_updates_per_iteration):
                    self.actor.train()
                    self.critic.train()
                    critic_loss_sum = 0
                    for step in range(self.max_timesteps_per_episode):
                        print("!!!!!!!!!!!!!!!!!!!!!!1",step)
                        mini_batch_obs = batch_obs[step::self.max_timesteps_per_episode,:step+1]
                        mini_batch_acts = batch_acts[step::self.max_timesteps_per_episode]
                        mini_batch_log_probs = batch_log_probs[step::self.max_timesteps_per_episode]
                        mini_batch_rtgs = batch_rtgs[step::self.max_timesteps_per_episode]
                        mini_A_k = A_k[step::self.max_timesteps_per_episode]
                        # print('mini_batch_obs.shape', mini_batch_obs.shape)
                        # print('mini_batch_acts.shape', mini_batch_acts.shape)
                        # print('mini_batch_log_probs.shape', mini_batch_log_probs.shape)
                        # print('mini_batch_rtgs.shape', mini_batch_rtgs.shape)
                        V, curr_log_probs = self.evaluate(mini_batch_obs, mini_batch_acts)
                        ratios = torch.exp(curr_log_probs - mini_batch_log_probs)
                        surr1 = ratios * mini_A_k
                        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * mini_A_k
                        actor_loss = -torch.min(surr1, surr2).mean()/self.max_timesteps_per_episode #+ perc_loss_actor.mean()
                        actor_loss.backward()
                        if step==self.max_timesteps_per_episode-1:
                            self.actor_optim.step()
                            self.actor_optim.zero_grad()

                        critic_loss = nn.MSELoss()(V, mini_batch_rtgs)/self.max_timesteps_per_episode #+ perc_loss_critic.mean()
                        critic_loss.backward() 
                        critic_loss_sum += critic_loss.item()
                        if step==self.max_timesteps_per_episode-1:  
                            self.critic_optim.step()
                            self.critic_optim.zero_grad()   
                    self.critic_logger.add_value(critic_loss.item())
                    self.critic_logger.add_value(critic_loss_sum)
                    self.critic_logger.plot()


torch.manual_seed('114514')
batch_size = 1
env = gym.make('EADEnv-v0',batch_size=batch_size, max_step=3)
print('env.observation_space.shape', env.observation_space.shape)
print('env.action_space.shape', env.action_space.shape)

ppo = PPO(env, batch_size=4)
ppo.learn(epoch_number=50)