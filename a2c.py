import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

from modules import FeatureEncoderNet, Storage


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

        actions_prob = F.softmax(policy_logits)
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

        self.ep_rewards = [0] * config['parallel_envs']
        self.ep_num = [0] * config['parallel_envs']

        num_actions = self.env.action_space.n
        self.actor_critic = ActorCritic(num_actions, config).to(device)
        self.optim = torch.optim.Adam(self.actor_critic.parameters(), lr=config['lr'])

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
            next_action, action_log_prob, entropy, value = self.actor_critic(self.process_obs(obs))
            obs, rewards, dones, infos = self.env.step(next_action.cpu().numpy())

            mask = (1 - torch.from_numpy(np.array(dones, dtype=np.float32))).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)

            # store values from current step
            rollout.add(next_action, action_log_prob, rewards, value, mask, entropy)

            # accumulate rewards over entire episode (until done)
            for env in range(len(rewards)):
                self.ep_rewards[env] += rewards[env].item()
                # log accumulated reward when an episode is done and reset
                if dones[env] > 0:
                    self.ep_num[env] += 1
                    self.writer.add_scalar('train/env{}_ep_reward'.format(env), self.ep_rewards[env], self.ep_num[env])
                    self.ep_rewards[env] = 0.0

            # reset feature extractor LSTM cell and hidden states
            self.actor_critic.reset_lstm(dones)

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
            actions, action_log_probs, values, rewards, masks, entropies = rollout.process()

            # collecting target for value network
            # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
            # append final value
            # rewards_plus_v = torch.cat([rewards, values[-1, :].detach().unsqueeze(0)], 0)
            # discount accumulates rewards for each rollout step
            discounted_rewards = self.discount(rewards, masks, final_value, gamma)

            # values is length n = num rollout_steps
            # each v_i is the predicted accumulated reward starting from the ith rollout_step
            advantages = discounted_rewards - values

            # action_log_probs = [4 x t x 1]
            policy_net_loss = (-action_log_probs * advantages.detach()).mean()
            value_net_loss = (advantages ** 2).mean() * self.config['value_beta']
            entropy = entropies.sum() * self.config['entropy_beta']

            loss = policy_net_loss + value_net_loss + entropy

            # Do Update Step for Model
            self.optim.zero_grad()
            loss.backward()
            # clip gradients for better stability
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
            self.optim.step()
            
            # =======================================================================================================
            # SAVE CHECKPOINTS
            # =======================================================================================================
            if loss < best_loss:
                best_loss = loss.item()
                torch.save(self.actor_critic.state_dict(), os.path.join(self.config['outdir'], 'model_best_ckpt'))

            if i % self.config['save_iters'] == 0:
                torch.save(self.actor_critic.state_dict(), os.path.join(self.config['outdir'], 'model_recent_ckpt'))

            # =======================================================================================================
            # LOGGING
            # =======================================================================================================
            if self.writer is not None and i % self.config['log_iters'] == 0:
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', loss.item(), i)
                    self.writer.add_scalar('train/policy_loss', policy_net_loss.item(), i)
                    self.writer.add_scalar('train/value_loss', value_net_loss.item(), i)
                    self.writer.add_scalar('train/entropy', entropy.item(), i)
                    self.writer.add_scalar('train/step_reward', rewards.sum().item(), i)

                output = "Update {}/{}\n".format(i, self.config['num_updates'])
                output += "loss: {:.4f}, ".format(loss.item())
                output += "policy_net_loss: {:.4f}, ".format(policy_net_loss.item())
                output += "value_net_loss: {:.4f}, ".format(value_net_loss.item())
                output += "entropy: {:.4f}, ".format(entropy.item())
                output += "step_reward: {:.4f}\n".format(rewards.sum().item())
                self.logger.info(output)
