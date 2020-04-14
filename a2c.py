import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import deque

from modules import FeatureEncoderNet, Storage
from icm import ICMAgent

class ActorCritic(nn.Module):
    def __init__(self, action_space_size, config):
        super(ActorCritic, self).__init__()

        features_size = 288 # 288 = (48/(2^4))^2 * 32 
        self.extract_feats = FeatureEncoderNet(
            buf_size=config['parallel_envs'],
            ch_in=config['state_frames'],
            conv_out_size=features_size, 
            lstm_hidden_size=features_size) # original paper used 256

        # actor is the policy network: pi(state) -> action
        self.actor = nn.Linear(features_size, action_space_size)

        # critic is the value network: V(state, action) -> value
        # that predicts the expected reward starting from current timestep
        self.critic = nn.Linear(features_size, 1)

    def forward(self, state):
        feature = self.extract_feats(state)
        policy_logits = self.actor(feature)
        value = self.critic(feature)

        actions_prob = F.softmax(policy_logits, dim=-1)
        actions_distr = torch.distributions.Categorical(actions_prob)
        next_action = actions_distr.sample()

        return next_action, actions_distr.log_prob(next_action), actions_distr.entropy().mean(), value.squeeze(1)

    def reset_lstm(self, reset_indices):
        self.extract_feats.reset_lstm(reset_indices)


class A2C(object):
    def __init__(self, config, env, device, logger, writer=None):
        self.config = config
        self.env = env
        self.device = device
        self.logger = logger
        self.writer = writer

        self.ep_rewards = deque(maxlen=4)

        self.num_actions = self.env.action_space.n
        self.actor_critic = ActorCritic(self.num_actions, config).to(device)
        self.optim = torch.optim.Adam(self.actor_critic.parameters(), lr=config['lr'])

        if self.config['use_icm']:
            self.icm_fwd_loss = []
            self.icm_inv_loss = []
            self.curiosity = ICMAgent(self.num_actions, self.config, self.device).to(self.device)
            # lr might have to be diff
            self.optim_icm = torch.optim.Adam(self.curiosity.parameters(), lr=config['lr'])

    def process_obs(self, obs):
        # 1. reorder dimensions for nn.Conv2d (batch, ch_in, width, height)
        # 2. convert numpy array to _normalized_ FloatTensor
        tensor = torch.from_numpy(obs.astype(np.float32).transpose((0, 3, 1, 2))) / 255
        # Resize spatial size to 48
        tensor = torch.nn.functional.interpolate(tensor, scale_factor=48/tensor.shape[-1])
        return tensor.to(self.device)
    
    def discount(self, x, masks, final_value, gamma):
        """
            x = [r1, r2, r3, ..., rN]
            returns [r1 + r2*gamma + r3*gamma^2 + ...,
                     r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                    ..., ..., rN]
        """
        num_steps = len(x)
        X = final_value
        out = torch.zeros((x.shape), device=self.device)
        for t in reversed(range(num_steps)):
            X = x[t] + X * gamma * masks[t]
            out[t] = X
        return out

    def run_episode(self, obs):
        rollout = Storage(self.config['parallel_envs'])
        
        for i in range(self.config['rollout_steps']):
            curr_state = obs
            next_action, action_log_prob, entropy, value = self.actor_critic(self.process_obs(obs))
            obs, rewards, dones, infos = self.env.step(next_action.cpu().numpy())

            mask = (1 - torch.from_numpy(np.array(dones, dtype=np.float32))).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)

            for info in infos:
                if 'episode' in info.keys():
                    self.ep_rewards.append(info['episode']['r'])
            
            if self.config['use_icm']:
                # ICM stuff here
                action = torch.zeros((4,self.num_actions), dtype=torch.float32, device=self.device)
                for i in range(4):
                    action[i,next_action[i]] = 1
                ireward, next_pred_state_features, pred_action, next_state_features = self.curiosity.forward(
                    action, 
                    self.process_obs(curr_state), 
                    self.process_obs(obs)
                    )
                fwd_model_loss = ((next_pred_state_features-next_state_features)**2).mean()
                inverse_model_loss = ((pred_action-action)**2).mean()
                self.icm_fwd_loss += [fwd_model_loss.item()*self.config['fwd_beta']]
                self.icm_inv_loss += [inverse_model_loss.item()*(1-self.config['fwd_beta'])]
                loss = fwd_model_loss*self.config['fwd_beta'] + inverse_model_loss*(1-self.config['fwd_beta'])
                # Do update step for ICM
                self.optim_icm.zero_grad()
                loss.backward()
                self.optim_icm.step()

                # store values from current step
                rollout.add(next_action, action_log_prob, rewards, value, mask, entropy, ireward)
            else:
                rollout.add(next_action, action_log_prob, rewards, value, mask, entropy)

            # reset feature extractor LSTM cell and hidden states
            self.actor_critic.reset_lstm(dones)


        # use final_value as prediction of final reward in discounted accumulation
        with torch.no_grad():
            _, _, _, final_value = self.actor_critic(self.process_obs(obs)) 

        return rollout, obs, final_value

    def train(self):
        obs = self.env.reset()
        gamma = self.config['gamma']
        best_loss = np.inf

        iter_range = range(1, self.config['num_updates']+1)
        if self.config['use_tqdm']:
            iter_range = tqdm(iter_range)

        for i in iter_range:
            rollout, obs, final_value = self.run_episode(obs)
            if self.config['use_icm']:
                actions, action_log_probs, values, ex_rewards, masks, entropies, i_rewards = rollout.process()
            else:
                actions, action_log_probs, values, ex_rewards, masks, entropies = rollout.process()

            # collecting target for value network
            # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
            # append final value
            # rewards_plus_v = torch.cat([rewards, values[-1, :].detach().unsqueeze(0)], 0)
            # discount accumulates rewards for each rollout step
            rewards = ex_rewards
            if self.config['use_icm']:
                rewards += i_rewards*self.config['pred_beta']
            discounted_rewards = self.discount(rewards, masks, final_value, gamma)

            # values is length n = num rollout_steps
            # each v_i is the predicted accumulated reward starting from the ith rollout_step
            advantages = discounted_rewards - values

            # action_log_probs = [4 x t x 1]
            policy_net_loss = (-action_log_probs * advantages.detach()).mean()
            value_net_loss = (advantages ** 2).mean() * self.config['value_beta']
            entropy = entropies.sum() * self.config['entropy_beta']

            loss = policy_net_loss + value_net_loss - entropy

            # Do Update Step for Model
            self.optim.zero_grad()
            loss.backward()
            # clip gradients for better stability
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
            self.optim.step()
            
            # =======================================================================================================
            # SAVE CHECKPOINTS
            # =======================================================================================================
            # if loss < best_loss:
            #     best_loss = loss.item()
            #     torch.save(self.actor_critic.state_dict(), os.path.join(self.config['outdir'], 'model_best_ckpt'))

            if i % self.config['save_iters'] == 0:
                torch.save(self.actor_critic.state_dict(), os.path.join(self.config['outdir'], 'model_recent_ckpt'))

            # =======================================================================================================
            # LOGGING
            # =======================================================================================================
            if self.writer is not None and i % self.config['log_iters'] == 0:
                ep_max_reward = np.max(self.ep_rewards) if len(self.ep_rewards) > 0 else 0.0
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', loss.item(), i)
                    self.writer.add_scalar('train/policy_loss', policy_net_loss.item(), i)
                    self.writer.add_scalar('train/value_loss', value_net_loss.item(), i)
                    self.writer.add_scalar('train/entropy', entropy.item(), i)
                    self.writer.add_scalar('train/ep_max_reward', ep_max_reward, i)
                    if self.config['use_icm']:
                        self.writer.add_scalar('train/i_reward',i_rewards.mean().item()*self.config['pred_beta'], i)
                        self.writer.add_scalar('train/ex_reward',ex_rewards.mean().item(), i)
                        self.writer.add_scalar('train/icm_fwd_loss',sum(self.icm_fwd_loss)/len(self.icm_fwd_loss), i)
                        self.writer.add_scalar('train/icm_inv_loss',sum(self.icm_inv_loss)/len(self.icm_inv_loss), i)

                output = "Update {}/{}\n".format(i, self.config['num_updates'])
                output += "loss: {:.4f}, ".format(loss.item())
                output += "policy_net_loss: {:.4f}, ".format(policy_net_loss.item())
                output += "value_net_loss: {:.4f}, ".format(value_net_loss.item())
                output += "entropy: {:.4f}, ".format(entropy.item())
                output += "ep_max_reward: {:.4f}\n".format(ep_max_reward)
                if self.config['use_icm']:
                    output += "i_reward: {:.4f}, ".format(i_rewards.mean().item()*self.config['pred_beta'])
                    output += "ex_reward: {:.4f}, ".format(ex_rewards.mean().item())
                    output += "icm_fwd_loss: {:.4f}, ".format(sum(self.icm_fwd_loss)/len(self.icm_fwd_loss))
                    output += "icm_inv_loss: {:.4f}\n".format(sum(self.icm_inv_loss)/len(self.icm_inv_loss))
                    self.icm_fwd_loss = []
                    self.icm_inv_loss = []
                self.logger.info(output)
