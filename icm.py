import torch
import torch.nn as nn
from modules import FeatureEncoderNet

class ICMAgent(nn.Module):
    def __init__(self, action_space_size, config, device):
        super(ICMAgent, self).__init__()
        features_size = 288 # same as ActorCritic
        self.device = device
        # feature network
        self.extract_feats = FeatureEncoderNet(
            buf_size=config['parallel_envs'],
            ch_in=config['state_frames'],
            conv_out_size=features_size, 
            lstm_hidden_size=features_size,
            use_lstm=False) # original paper used 256
        '''
        Forward Model from paper
        f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
        f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
        '''
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(features_size+action_space_size, features_size),
            torch.nn.ReLU(),
            torch.nn.Linear(features_size, features_size)
        )
        '''
        Inverse Model from paper
        g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
        logits = linear(g, ac_space, "glast", normalized_columns_initializer(0.01))
        '''
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(features_size*2, features_size),
            torch.nn.ReLU(),
            torch.nn.Linear(features_size, action_space_size)
        )
        
    def forward(self, action, curr_state, next_state):
        # output will be next predicted state & action, and intrinsic reward
        # get features from both states (phi st and phi st+1 from paper)
        curr_state_features = self.extract_feats(curr_state)
        next_state_features = self.extract_feats(next_state)
        # forward model next predicted state
        next_pred_state_features = self.forward_model(torch.cat([action,curr_state_features.detach()], dim=1))
        pred_action = self.inverse_model(torch.cat([curr_state_features, next_state_features], dim=1))
        with torch.no_grad():
            reward = torch.zeros(next_state_features.shape[0],device=self.device,dtype=torch.float32)
            for i in range(reward.shape[0]):
                reward[i] = ((next_pred_state_features[i]-next_state_features[i])**2).mean()
        return reward, next_pred_state_features, pred_action, next_state_features

