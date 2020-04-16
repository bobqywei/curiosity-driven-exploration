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
        with torch.no_grad():
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

        if self.config['use_icm']:
            self.icm_fwd_loss = []
            self.icm_inv_loss = []
            self.curiosity = ICMAgent(self.num_actions, self.config, self.device).to(self.device)
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            self.optim = torch.optim.Adam(
                list(self.actor_critic.parameters()) + list(self.curiosity.parameters()), 
                lr=config['lr'])
        
        else:
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
        """
        Runs a single episode of length "rollout_steps"
        """
        # storage for current episode
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

                # turn next action index into one hot vector
                one_hot_action = torch.zeros((self.config['parallel_envs'], self.num_actions), dtype=torch.float32, device=self.device)
                for i in range(self.config['parallel_envs']):
                    one_hot_action[i, next_action[i]] = 1

                # get intrinsic reward and outputs from ICM forward and inverse models
                ireward, next_pred_state_features, pred_action_logits, next_state_features = self.curiosity(
                    one_hot_action, self.process_obs(curr_state), self.process_obs(obs))

                # MSE between predicted next state and actual next state
                self.icm_fwd_loss.append(
                    ((next_pred_state_features - next_state_features) ** 2).mean())
                self.icm_inv_loss.append(
                    self.cross_entropy_loss(pred_action_logits, next_action))

                rollout.add(next_action, action_log_prob, rewards, value, mask, entropy, ireward)
            
            else:
                rollout.add(next_action, action_log_prob, rewards, value, mask, entropy)

            # reset feature extractor LSTM cell and hidden states
            self.actor_critic.reset_lstm(dones)

        # use final_value as prediction of final reward in discounted accumulation
        # no need for gradients for this final value
        with torch.no_grad():
            _, _, _, final_value = self.actor_critic(self.process_obs(obs)) 

        return rollout, obs, final_value

    def train(self):
        obs = self.env.reset()

        iter_range = range(1, self.config['num_updates']+1)
        if self.config['use_tqdm']:
            iter_range = tqdm(iter_range)

        for i in iter_range:
            # run single episode
            rollout, obs, final_value = self.run_episode(obs)

            # unpack episode storage
            if self.config['use_icm']:
                _, action_log_probs, values, ex_rewards, masks, entropies, i_rewards = rollout.process()
            else:
                _, action_log_probs, values, ex_rewards, masks, entropies = rollout.process()

            # combine extrinsic and intrinsic rewards
            rewards = ex_rewards
            if self.config['use_icm']:
                rewards += i_rewards * self.config['pred_beta']
            # discount and accumulate rewards (including final value)
            discounted_rewards = self.discount(rewards, masks, final_value, self.config['gamma'])
            # values is length n = num rollout_steps
            # each v_i is the predicted accumulated reward starting from the ith rollout_step
            advantages = discounted_rewards - values

            # action_log_probs = [4 x t x 1]
            policy_net_loss = (-action_log_probs * advantages.detach()).mean()
            value_net_loss = (advantages ** 2).mean() * self.config['value_beta']
            entropy = entropies.sum() * self.config['entropy_beta']

            if self.config['use_icm']:
                fwd_loss = sum(self.icm_fwd_loss) / len(self.icm_fwd_loss) # mean over list
                inv_loss = sum(self.icm_inv_loss) / len(self.icm_inv_loss) # mean over list
                self.icm_fwd_loss = []
                self.icm_inv_loss = []

            loss = policy_net_loss + value_net_loss - entropy + (fwd_loss + inv_loss)

            # Do Update Step for Model
            self.optim.zero_grad()
            loss.backward()
            # clip gradients for better stability
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
            self.optim.step()
            
            # =======================================================================================================
            # SAVE CHECKPOINTS
            # =======================================================================================================
            if i % self.config['save_iters'] == 0:
                torch.save(self.actor_critic.state_dict(), os.path.join(self.config['outdir'], 'actor_critic_ckpt_{}'.format(i)))
                torch.save(self.curiosity.state_dict(), os.path.join(self.config['outdir'], 'icm_ckpt_{}'.format(i)))

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
                        self.writer.add_scalar('train/icm_fwd_loss',fwd_loss.item(), i)
                        self.writer.add_scalar('train/icm_inv_loss',inv_loss.item(), i)

                output = "Update {}/{}\n".format(i, self.config['num_updates'])
                output += "loss: {:.4f}, ".format(loss.item())
                output += "policy_net_loss: {:.4f}, ".format(policy_net_loss.item())
                output += "value_net_loss: {:.4f}, ".format(value_net_loss.item())
                output += "entropy: {:.4f}, ".format(entropy.item())
                output += "ep_max_reward: {:.4f}, ".format(ep_max_reward)

                if self.config['use_icm']:
                    output += "i_reward: {:.4f}, ".format(i_rewards.mean().item()*self.config['pred_beta'])
                    output += "icm_fwd_loss: {}, ".format(fwd_loss.item())
                    output += "icm_inv_loss: {}\n".format(inv_loss.item())
                self.logger.info(output)
