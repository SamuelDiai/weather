import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor
from torch.autograd import Variable
import numpy as np

class naive_last_time_step(nn.Module):

    def __init__(self, idx_target):
        super(naive_last_time_step, self).__init__()
        self.idx_target = idx_target

    def forward(self, inputs):
        return inputs[:, -1, self.idx_target]

class naive_rolling_average(nn.Module):
    def __init__(self, idx_target):
        super(naive_rolling_average, self).__init__()
        self.idx_target = idx_target

    def forward(self, inputs):
        return inputs[:, :, self.idx_target].mean(axis = 1)

class model_LSTM(nn.Module):
    def __init__(self, input_size, n_hidden, num_layers, dropout):
        super(model_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, n_hidden, num_layers=num_layers, batch_first = True, dropout = dropout)
        self.lin = nn.Linear(n_hidden, 1)

    def forward(self, inputs):
        y, (h_n, c_n) = self.lstm(inputs)
        return self.lin(y[:, -1, :])

def init_hidden(x, hidden_size: int, device):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    # Variable is deprecated
    #return Variable(torch.zeros(1, x.size(0), hidden_size)).to(device)
    return torch.zeros(1, x.size(0), hidden_size).to(device)


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, T: int, device):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.device = device

        self.lstm_layer = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        self.linear_concat = nn.Linear(in_features = 2 * hidden_size,
                                                   out_features = T - 1)
        self.linear_data = nn.Linear(in_features = T - 1,
                                     out_features = T - 1)
        self.linear_attn_output = nn.Linear(in_features = T - 1,
                                            out_features = 1)

    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)
        #input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        #input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        input_weighted = torch.zeros(input_data.size(0), self.T - 1, self.input_size).to(self.device)
        input_encoded = torch.zeros(input_data.size(0), self.T - 1, self.hidden_size).to(self.device)
        weight = torch.zeros(input_data.size(0), self.T - 1, self.input_size).to(self.device)

        # hidden, state: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size, self.device)  # 1 * batch_size * hidden_size
        state = init_hidden(input_data, self.hidden_size, self.device)

        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            #x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
            #               state.repeat(self.input_size, 1, 1).permute(1, 0, 2),
            #               input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)

            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           state.repeat(self.input_size, 1, 1).permute(1, 0, 2)),
                           dim=2)
            # Eqn. 8: Get attention weights
            t1 = self.linear_concat(x)
            t2 = self.linear_data(input_data.permute(0, 2, 1))
            x = t1 + t2
            x = torch.tanh(x)
            x = self.linear_attn_output(x)
            # Eqn. 9: Softmax the attention weights
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=1)
            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, state))
            hidden = lstm_states[0]
            state = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
            weight[:, t, :] = attn_weights

        return input_weighted, input_encoded, weight


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, device, out_feats=1):
        super(Decoder, self).__init__()
        self.device = device
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, ( T - 1))
        # Initialize hidden and state, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size, self.device)
        state = init_hidden(input_encoded, self.decoder_hidden_size, self.device)
        #context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))
        context = torch.zeros(input_encoded.size(0), self.encoder_hidden_size).to(self.device)
        weight = torch.zeros(input_encoded.size(0), self.T - 1, self.T - 1).to(self.device)
        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           state.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = F.softmax(self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)
            weight[:, t, :] = x
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, state))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            state = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn. 22: final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1)), weight


class DA_RNN(nn.Module):
    def __init__(self,  input_size: int, encoder_hidden_size: int, decoder_hidden_size: int, T: int, idx_target, device, out_feats=1):
        super(DA_RNN, self).__init__()
        self.device = device
        self.idx_target = torch.LongTensor([idx_target]).to(device)
        self.mask = torch.LongTensor(np.array([i for i in range(input_size+1) if i != self.idx_target])).to(self.device)
        self.encoder = Encoder(input_size, encoder_hidden_size, T, self.device).to(self.device)
        self.decoder = Decoder(encoder_hidden_size, decoder_hidden_size, T, self.device, out_feats=1).to(self.device)

    def forward(self, inputs):
        y_history = inputs[:, :, self.idx_target]
        X = inputs[:, :, self.mask]
        input_weighted, input_encoded, spatial_weights = self.encoder(X)
        output, temporal_weights = self.decoder(input_encoded, y_history)
        return output

    def return_weights(self, inputs):
        with torch.no_grad():
            y_history = inputs[:, :, self.idx_target]
            X = inputs[:, :, self.mask]
            input_weighted, input_encoded, spatial_weights = self.encoder(X)
            output, temporal_weights = self.decoder(input_encoded, y_history)
        return spatial_weights, temporal_weights

class Causal_Conv_1d(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, dilation):
    super(Causal_Conv_1d, self).__init__()
    self.padding = kernel_size//2 * dilation
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride = 1, padding = self.padding, dilation=dilation).to(device)

  def forward(self, x):
    x = self.conv(x)
    return x[:, :, :-self.padding]

class Conv_Model(nn.Module):

    def __init__(self, input_size, n_past):
        super(Conv_Model, self).__init__()
        self.input_size = input_size
        self.n_past = n_past

        self.conv = nn.Sequential(
            Causal_Conv_1d(self.input_size, 16, kernel_size=2, dilation=1),
            nn.BatchNorm1d(num_features=16),
            nn.LeakyReLU(),
            Causal_Conv_1d(16, 32, kernel_size=2, dilation=2),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(),
            Causal_Conv_1d(32, 64, kernel_size=2, dilation=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            Causal_Conv_1d(64, 128, kernel_size=2, dilation=8),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            Causal_Conv_1d(128, 256, kernel_size=2, dilation=16),
            nn.BatchNorm1d(256),
        )

        self.lin = nn.Sequential(
        # nn.Linear(16*self.n_past, 256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # x = x.view(batch_size, -1)
        x = x[:, :, -1]
        x = self.lin(x.view(batch_size, -1))
        return x
