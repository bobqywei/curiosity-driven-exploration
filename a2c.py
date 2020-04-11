import torch
import torch.nn as nn
import numpy as np
import scipy.signal
import torch.nn.functional as F

from modules import FeatureEncoderNet


def discount(x, masks, gamma):
    """
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                   r2 + r3*gamma + r4*gamma^2 + ...,
                     r3 + r4*gamma + r5*gamma^2 + ...,
                        ..., ..., rN]
    """
    num_steps = len(x)
    X = x[-1]
    out = torch.zeros((x.shape), device=)
    for t in reversed(range(num_steps-1)):
        X = x[t] + X * gamma * masks[t]
        out[t] = X
    return torch.cat(out, 0)


class Storage(object):
    def __init__(self, config):
        self.num_envs = config['parallel_envs']

        self.states = [] # N - 4 x M tensors
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        # self.features = []

    def add(self, state, action, reward, value):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        # self.features += [features]

    def extend(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        # self.features.extend(other.features)

    def process(self):
        return map(lambda x: torch.cat(x, 0), [...])


class ActorCritic(nn.Module):
    def __init__(self, action_space_size):
        super(self, ActorCritic).__init__()
        features_size = 288 # 288 = (48/(2^4))^2 * 32 
        self.extract_feats = FeatureEncoderNet(
            batch_size=config['parallel_envs'], 
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

        return next_action, actions_distr.log_prob(next_action), actions_distr.entropy.mean(), value

    def reset_lstm(self, reset_indices):
        self.extract_feats.reset_lstm(reset_indices)


class A2C(object):
    def __init__(self, config, env, device):
        self.config = config
        self.env = env
        self.device = device

        num_actions = self.env.action_space.n
        
        self.actor_critic = ActorCritic(num_actions)
        self.optim = torch.nn.optim.Adam(self.actor_critic.parameters())

    def run_episode(self, obs):
        rollout = Storage()
        
        for i in range(self.config['rollout_steps']):
            next_action, action_log_prob, entropy, value = self.actor_critic(obs)
            obs, rewards, dones, infos = env.step(next_action.cpu().numpy())

            # reset feature extractor LSTM cell and hidden states
            self.actor_critic.reset_lstm(dones)

            masks = (1 - torch.from_numpy(np.array(dones, dtype=np.float32))).to(self.device).unsqueeze(1)

            rollout.add(...)

            if i == len(self.config['rollout_steps']) - 1:
                rollout.add(final_value)

    def train(self):
        obs = self.env.reset()
        gamma = self.config['gamma']
        lamb = self.config['lambd']

        # Need optimizers

        for i in range(self.config['num_updates']):
            rollout = run_episode(obs)
            actions, action_log_probs, values, rewards, masks, final_value, entropies = rollout.process()

            # collecting target for value network
            # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
            rewards_plus_v = torch.cat([rewards, final_value.unsqueeze(0)], 0)
            discounted_rewards = discount(rewards_plus_v, masks, gamma)[:-1]

            values = torch.cat([values, final_value.unsqueeze(0)], 0)
            delta_t = rewards + gamma * values[1:] - values[:-1]
            # TODO: Need to fix discount so that it returns same size tensor back
            advantages = discount(delta_t, gamma * lambd)

            # action_log_probs = [4 x t x 1]
            policy_net_loss = (-action_log_probs * advantages.detach()).sum()
            # BUG
            value_net_loss = (0.5 * (values - discounted_rewards) ** 2).sum()
            # TODO: need to reconsider which values need to be masked
            entropy_loss = entropies.sum()

            loss = policy_net_loss + value_net_loss * self.config['value_beta'] + entropy_loss * self.config['entropy_beta']

            # Do Update Step for Model
            optim.zero_grad()
            loss.backward()
            optim.step()
        