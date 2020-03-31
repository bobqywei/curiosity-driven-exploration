import torch
import torch.nn as nn
import numpy as np


class FeatureEncoderNet(nn.Module):
    def __init__(self, batch_size, ch_in, conv_out_size, lstm_hidden_size, use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.conv_out_size = conv_out_size

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
            self.lstm_hidden_state = self.lstm_cell_state = torch.zeros(
                (batch_size, lstm_hidden_size), 
                device=self.lstm.weight_ih.device)

    def reset_lstm(self, reset_indices):
        if self.use_lstm:
            with torch.no_grad():
                resetTensor = torch.as_tensor(reset_indices.astype(np.uint8), device=self.lstm.weight_ih.device)
                if resetTensor.sum():
                    self.lstm_hidden_state = (1 - resetTensor.view(-1, 1)).float() * self.lstm_hidden_state
                    self.lstm_cell_state = (1 - resetTensor.view(-1, 1)).float() * self.lstm_cell_state

    def forward(self, raw_state):
        x = self.conv(raw_state)

        if self.use_lstm:
            x = x.view(-1, self.conv_out_size)
            self.lstm_hidden_state, self.lstm_cell_state = self.lstm(x, (self.lstm_hidden_state, self.lstm_cell_state))
            return self.lstm_hidden_state

        else:
            return x.view(-1, self.conv_out_size)