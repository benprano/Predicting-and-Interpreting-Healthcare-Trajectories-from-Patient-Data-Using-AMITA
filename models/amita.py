import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def seed_all(seed: int = 1992):
    """Seed all random number generators."""
    print("Using Seed Number {}".format(seed))

    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # set PYTHONHASHSEED env var at fixed value
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # pytorch (both CPU and CUDA)
    np.random.seed(seed)  # for numpy pseudo-random generator
    # set fixed value for python built-in pseudo-random generator
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


seed_all()


class AmitaLSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size, output_dim, batch_first=True, bidirectional=True):
        super(AmitaLSTM, self).__init__()
        self.input_size = input_size
        self.output_dim = output_dim
        self.initializer_range = 0.02
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([self.input_size, 1, self.hidden_size]).float()
        self.decay_features = torch.Tensor(torch.arange(self.input_size)).float()
        self.register_buffer('c1_const', self.c1)
        self.register_buffer('c2_const', self.c2)
        self.register_buffer("ones_const", self.ones)
        self.alpha = torch.FloatTensor([0.5])
        self.alpha_imp = torch.FloatTensor([0.5])
        self.register_buffer("factor", self.alpha)
        self.register_buffer("features_decay", self.decay_features)
        self.register_buffer("factor_impu", self.alpha_imp)

        self.U_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.U_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))
        self.Dw = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1, self.hidden_size)))

        self.W_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))
        self.W_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size, self.hidden_size)))

        self.W_cell_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.W_cell_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        self.b_decomp = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_j = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_i = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_f = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_o = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_c = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_last = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_time = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
        self.b_d = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))

        # Gate Linear Unit for last records
        self.activation_layer = nn.ELU()

        self.F_alpha_n = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size,
                                                                                      self.hidden_size * 2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, 1)))
        self.F_beta = nn.Linear(4 * self.hidden_size, 1)
        self.Phi = nn.Linear(4 * self.hidden_size, self.output_dim)

    @torch.jit.script_method
    def amita_unit(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x = inputs
        t = times
        lst = last_data
        freq = freq_list
        T = self.map_elapse_time(t)
        D_ST = torch.tanh(torch.einsum("bij,ijk->bik", c_tilda_t, self.W_decomp)) # Short-term memory contribution
        # Apply temporal decay to D-STM
        decay_factor = torch.mul(T, self.freq_decay(freq, h_tilda_t))
        D_ST_decayed = D_ST * decay_factor
        LTM = c_tilda_t - D_ST + D_ST_decayed  # Long-term memory contribution
        # Combine short-term and long-term memory
        c_tilda_t = D_ST_decayed + LTM
        # Last observation
        last_tilda_t = self.activation_layer(torch.einsum("bij,jik->bjk", lst.unsqueeze(1), 
                                                          self.U_last) + self.b_last)
        # Ajust previous to incoporate the latest records for each feature
        h_tilda_t = h_tilda_t + last_tilda_t

        # Capturing Temporal Dependencies wrt to the previous hidden state
        j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +
                               torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_j) + self.b_j)

        # Time Gate
        t_gate = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_time) +
                               torch.sigmoid(self.map_elapse_time(t)) + self.b_time)
        # Input Gate
        i = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_i) +
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) +
                          c_tilda_t * self.W_cell_i + self.b_i * self.freq_decay(freq, h_tilda_t))
        # Forget Gate
        f = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_f) +
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) +
                          c_tilda_t * self.W_cell_f + self.b_f + h_tilda_t)

        f_new = f * self.map_elapse_time(t) + (1 - f) * self.freq_decay(freq, h_tilda_t)
        # Candidate Memory Cell
        C = torch.tanh(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_c) +
                       torch.einsum("bij,ijk->bik", h_tilda_t, self.W_c) + self.b_c)
        # Current Memory Cell
        Ct = (f_new + t_gate) * c_tilda_t + i * j_tilda_t * t_gate * C
        # Output Gate
        o = torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_o) +
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) +
                          t_gate + last_tilda_t + Ct * self.W_cell_o + self.b_o)
        # Current Hidden State
        h_tilda_t = o * torch.tanh(Ct + last_tilda_t)

        return h_tilda_t, Ct, self.freq_decay(freq, h_tilda_t), f_new

    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def freq_decay(self, freq_dict: torch.Tensor, ht: torch.Tensor):
        freq_weight = torch.exp(-self.factor_impu * freq_dict)
        weights = torch.sigmoid(torch.einsum("bij,jik->bjk", freq_weight.unsqueeze(-1), self.Dw) + \
                                torch.einsum("bij,ijk->bik", ht, self.W_d) + self.b_d)
        return weights

    @torch.jit.script_method
    def forward(self, inputs, times, last_values, freqs):
        device = inputs.device
        if self.batch_first:
            batch_size = inputs.size()[0]
            inputs = inputs.permute(1, 0, 2)
            last_values = last_values.permute(1, 0, 2)
            freqs = freqs.permute(1, 0, 2)
            times = times.transpose(0, 1)
        else:
            batch_size = inputs.size()[1]
        prev_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
        prev_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)

        seq_len = inputs.size()[0]
        hidden_his = torch.jit.annotate(List[Tensor], [])
        weights_decay = torch.jit.annotate(List[Tensor], [])
        weights_fgate = torch.jit.annotate(List[Tensor], [])
        for i in range(seq_len):
            prev_hidden, prev_cell, pre_we_decay, fgate_f = self.amita_unit(prev_hidden, prev_cell,
                                                                            inputs[i], times[i],
                                                                            last_values[i], freqs[i])
            hidden_his += [prev_hidden]
            weights_decay += [pre_we_decay]
            weights_fgate += [fgate_f]
        hidden_his = torch.stack(hidden_his)
        weights_decay = torch.stack(weights_decay)
        weights_fgate = torch.stack(weights_fgate)
        if self.bidirectional:
            second_hidden = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_cell = torch.zeros((batch_size, inputs.size()[2], self.hidden_size), device=device)
            second_inputs = torch.flip(inputs, [0])
            second_times = torch.flip(times, [0])
            second_hidden_his = torch.jit.annotate(List[Tensor], [])
            second_weights_decay = torch.jit.annotate(List[Tensor], [])
            second_weights_fgate = torch.jit.annotate(List[Tensor], [])
            for i in range(seq_len):
                if i == 0:
                    time = times[i]
                else:
                    time = second_times[i - 1]
                second_hidden, second_cell, b_we_decay, fgate_b = self.amita_unit(second_hidden, second_cell,
                                                                                  second_inputs[i], time,
                                                                                  last_values[i], freqs[i])
                second_hidden_his += [second_hidden]
                second_weights_decay += [b_we_decay]
                second_weights_fgate += [fgate_b]
            second_hidden_his = torch.stack(second_hidden_his)
            second_weights_fgate = torch.stack(second_weights_fgate)
            second_weights_decay = torch.stack(second_weights_decay)
            weights_decay = torch.cat((weights_decay, second_weights_decay), dim=-1)
            weights_fgate = torch.cat((weights_fgate, second_weights_fgate), dim=-1)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=-1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=-1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 2, 3)
            weights_fgate = weights_fgate.permute(1, 0, 2, 3)

        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * hidden_his, dim=1)
        hg = torch.cat([g_n, prev_hidden], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        # Max pooling
        betas = betas / torch.max(betas, dim=1, keepdim=True).values
        outs = torch.sum(betas * mu, dim=1)
        return outs, alphas, betas, weights_decay, weights_fgate


class Amita(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(Amita, self).__init__()
        # hidden dimensions
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_dim = output_dim
        # Temporal embedding AMITA LSTM
        self.amita_lstm = AmitaLSTM(self.input_size, self.hidden_size, self.output_dim)

    def forward(self, historic_features, timestamp, last_features, features_freqs, is_test=False):
        # Temporal features embedding
        outputs, alphas, betas, decay_weights, fgate = self.amita_lstm(historic_features, timestamp,
                                                                       last_features, features_freqs)
        if is_test:
            return alphas, betas, decay_weights, fgate, outputs
        else:
            return outputs
