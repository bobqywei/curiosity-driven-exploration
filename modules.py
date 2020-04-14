import torch
import torch.nn as nn
import numpy as np


class Storage(object):
    def __init__(self, num_envs):
        self.num_envs = num_envs

        # self.states = [] # N - 4 x M tensors
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        # self.final_value = 0.0
        self.entropies = []
        self.masks = []
        self.irewards = [] # ICM rewards
        # self.features = []

    def add(self, action, action_log_prob, reward, value, mask, entropy, ireward=None):
        # self.states += [state]
        self.actions += [action]
        self.action_log_probs += [action_log_prob]
        self.rewards += [reward]
        self.values += [value]
        self.masks += [mask]
        self.entropies += [entropy]
        if ireward is not None:
            self.irewards += [ireward]
        # self.features += [features]

    def process(self):
        return map(lambda x: torch.cat([t.unsqueeze(0) for t in x], 0), filter(lambda x: len(x) > 0,[
            self.actions, 
            self.action_log_probs, 
            self.values, 
            self.rewards, 
            self.masks, 
            self.entropies,
            self.irewards]))


class FeatureEncoderNet(nn.Module):
    def __init__(self, buf_size, ch_in, conv_out_size, lstm_hidden_size, use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.conv_out_size = conv_out_size
        self.lstm_hidden_size = lstm_hidden_size

        ch_out = 32
        filter_size = 3
        stride = 2
        padding = filter_size//2
        num_conv = 4

        # Initialize CNN to process raw image input
        conv = [nn.Conv2d(ch_in, ch_out, filter_size, stride, padding),
                nn.LeakyReLU(inplace=True)]
        for _ in range(num_conv-1):
            conv.append(nn.Conv2d(ch_out, ch_out, filter_size, stride, padding))
            conv.append(nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        if self.use_lstm:
            self.lstm = nn.LSTMCell(input_size=self.conv_out_size, hidden_size=lstm_hidden_size)

            with torch.no_grad():
                self._lstm_hidden_state = self._lstm_cell_state = torch.zeros(
                    (buf_size, lstm_hidden_size),
                    device=self.lstm.weight_ih.device)       

    def reset_lstm(self, reset_indices):
        if self.use_lstm:
            with torch.no_grad():
                resetTensor = torch.as_tensor(reset_indices.astype(np.uint8), device=self.lstm.weight_ih.device)
                if resetTensor.sum():
                    self._lstm_hidden_state = (1 - resetTensor.view(-1, 1)).float() * self._lstm_hidden_state
                    self._lstm_cell_state = (1 - resetTensor.view(-1, 1)).float() * self._lstm_cell_state
                else:
                    self._lstm_hidden_state = self._lstm_hidden_state.detach()
                    self._lstm_cell_state = self._lstm_cell_state.detach()

    def forward(self, raw_state):
        x = self.conv(raw_state)

        if self.use_lstm:
            x = x.view(-1, self.conv_out_size)
            self._lstm_hidden_state, self._lstm_cell_state = self.lstm(x, (self._lstm_hidden_state, self._lstm_cell_state))
            return self._lstm_hidden_state

        else:
            return x.view(-1, self.conv_out_size)

    def _apply(self, fn):
        super(FeatureEncoderNet, self)._apply(fn)
        if self.use_lstm:
            self._lstm_hidden_state = fn(self._lstm_hidden_state)
            self._lstm_cell_state = fn(self._lstm_cell_state)
        return self