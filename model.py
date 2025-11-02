# Import common packages
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.distributions import Gamma
import math

class EIRecLinear(nn.Module):
    r"""Recurrent E-I Linear transformation.
    
    Args:
        hidden_size: int, layer size
        e_prop: float between 0 and 1, proportion of excitatory units
    """
    __constants__ = ['bias', 'hidden_size', 'e_prop']

    def __init__(self, hidden_size, e_prop, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.e_prop = e_prop
        self.e_size = int(e_prop * hidden_size)
        self.i_size = hidden_size - self.e_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        sign_mask  = np.tile([1]*self.e_size+[-1]*self.i_size, (hidden_size, 1))
        np.fill_diagonal(sign_mask , 0)
        self.register_buffer('sign_mask', torch.from_numpy(sign_mask).float())

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            gamma_e = Gamma(0.1, 1)
            gamma_i = Gamma(0.2, 1)
            w = torch.empty_like(self.weight)
            w[:, :self.e_size] = gamma_e.sample((self.hidden_size, self.e_size))
            w[:, self.e_size:] = gamma_i.sample((self.hidden_size, self.i_size))
            w[:, :self.e_size] /= (self.e_size/self.i_size)
            self.weight.data.copy_(w)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def effective_weight(self):
        return torch.abs(self.weight) * self.sign_mask

    def forward(self, input):
        return F.linear(input, self.effective_weight(), self.bias) # weight is non-negative


class EIRNN(nn.Module):
    """Excitatory-Inhibitory RNN model with biologically inspired constraints."""

    def __init__(self, input_size, hidden_size, dt=None,
                 e_prop=0.8, sigma_rec=0.05, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.e_size = int(hidden_size * e_prop)
        self.i_size = hidden_size - self.e_size
        self.tau_r = 100
        self.alpha_r = dt / self.tau_r
        self.oneminusalpha_r = 1 - self.alpha_r
        self._sigma_rec = np.sqrt(2 * self.alpha_r) * sigma_rec

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = EIRecLinear(hidden_size, e_prop=0.8)

    def init_hidden(self, input):
        batch_size = input.shape[1]
        return (torch.zeros(batch_size, self.hidden_size).to(input.device),
                torch.zeros(batch_size, self.hidden_size).to(input.device))

    def recurrence(self, input, hidden):
        """Compute recurrence step."""
        state, output = hidden
        total_input = self.input2h(input) + self.h2h(output)
        state = state * self.oneminusalpha_r + total_input * self.alpha_r
        state += self._sigma_rec * torch.randn_like(state)
        output = torch.sigmoid(state)
        return state, output

    def forward(self, input, hidden=None):
        """Propagate input through the network."""
        if hidden is None:
            hidden = self.init_hidden(input)
        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden[1])
        output = torch.stack(output, dim=0)
        return output, hidden

class Net(nn.Module):
    """Recurrent network model with E-I constraints."""
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.05, **kwargs):
        super().__init__()
        self.rnn = EIRNN(input_size, hidden_size, **kwargs)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(self.rnn.e_size, output_size)
    def forward(self, x):
        rnn_activity, _ = self.rnn(x)
        rnn_e = rnn_activity[:, :, :self.rnn.e_size]
        dropped = self.dropout(rnn_e)
        out = self.fc(dropped)
        return out, rnn_activity