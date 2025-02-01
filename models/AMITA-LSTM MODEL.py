import torch
import torch.nn as nn
import random
import pandas as pd
from copy import deepcopy
from typing import Dict
import transformers
from torch import Tensor
from torch.nn import init, Parameter
import torch.nn.functional as F
import pdb
import math
import numpy as np
from sklearn import metrics
import os 
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, f1_score, average_precision_score
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

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


class AMITA_LSTM(torch.jit.ScriptModule):
    def __init__(self, input_size, hidden_size,output_dim, batch_first=True, bidirectional=True):
        super(AMITA_LSTM, self).__init__()
        self.input_size = input_size
        self.output_dim = output_dim
        self.initializer_range=0.02
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.c1 = torch.Tensor([1]).float()
        self.c2 = torch.Tensor([np.e]).float()
        self.ones = torch.ones([self.input_size,1, self.hidden_size]).float()
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
        self.W_cell_f= nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size, self.hidden_size)))
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
        
        #Gate Linear Unit for last records
        self.activation_layer = nn.ELU()
        
        self.F_alpha_n = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size,
                                                                                      self.hidden_size*2, 1)))
        self.F_alpha_n_b = nn.Parameter(torch.normal(0.0, self.initializer_range, size=(self.input_size,1)))
        self.F_beta = nn.Linear(4*self.hidden_size, 1)
        self.Phi = nn.Linear(4*self.hidden_size, self.output_dim)
    @torch.jit.script_method    
    def TLSTM_unit(self, prev_hidden_memory, cell_hidden_memory, inputs, times, last_data, freq_list):
        h_tilda_t, c_tilda_t = prev_hidden_memory, cell_hidden_memory,
        x = inputs
        t = times
        l = last_data
        freq=freq_list
        T = self.map_elapse_time(t)
        
        D_ST = torch.tanh(torch.einsum("bij,ijk->bik", c_tilda_t, self.W_decomp))  # Short-term memory contribution
        # Apply temporal decay to D-STM
        decay_factor = torch.mul(T, self.freq_decay(freq, h_tilda_t))
        D_ST_decayed = D_ST * decay_factor
        LTM = c_tilda_t - D_ST + D_ST_decayed  # Long-term memory contribution

        # Combine short-term and long-term memory
        c_tilda_t = D_ST_decayed + LTM
        
        last_tilda_t = self.activation_layer(torch.einsum("bij,jik->bjk", l.unsqueeze(1), 
                                                          self.U_last)+self.b_last)
        # Ajust previous to incoporate the latest records for each feature
        h_tilda_t = h_tilda_t + last_tilda_t
    
        # Capturing Temporal Dependencies wrt to the previous hidden state
        j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) +\
                               torch.einsum("bij,jik->bjk", x.unsqueeze(1),self.U_j) + self.b_j)
        
        # Time Gate
        t_gate = torch.sigmoid(torch.einsum("bij,jik->bjk",x.unsqueeze(1), self.U_time) + 
                               torch.sigmoid(self.map_elapse_time(t)) + self.b_time)
        # Input Gate
        i= torch.sigmoid(torch.einsum("bij,jik->bjk",x.unsqueeze(1), self.U_i)+\
                         torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i)+\
                         c_tilda_t*self.W_cell_i + self.b_i*self.freq_decay(freq, h_tilda_t))
        # Forget Gate
        f= torch.sigmoid(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_f)+\
                         torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f)+\
                         c_tilda_t*self.W_cell_f + self.b_f+j_tilda_t)

        f_new = f * self.map_elapse_time(t) + (1 - f) *  self.freq_decay(freq, h_tilda_t)
        # Candidate Memory Cell
        C =torch.tanh(torch.einsum("bij,jik->bjk", x.unsqueeze(1), self.U_c)+\
                      torch.einsum("bij,ijk->bik", h_tilda_t, self.W_c) + self.b_c)
        # Current Memory Cell
        Ct = (f_new + t_gate) * c_tilda_t + i * j_tilda_t * t_gate * C
        # Output Gate        
        o = torch.sigmoid(torch.einsum("bij,jik->bjk",x.unsqueeze(1), self.U_o)+
                          torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o)+
                          t_gate + last_tilda_t + Ct*self.W_cell_o + self.b_o)
        # Current Hidden State
        h_tilda_t = o * torch.tanh(Ct+last_tilda_t)
        
        return h_tilda_t, Ct, self.freq_decay(freq, h_tilda_t), f_new
    
    @torch.jit.script_method
    def map_elapse_time(self, t):
        T = torch.div(self.c1_const, torch.log(t + self.c2_const))
        T = torch.einsum("bij,jik->bjk", T.unsqueeze(1), self.ones_const)
        return T

    @torch.jit.script_method
    def freq_decay(self, freq_dict: torch.Tensor, ht: torch.Tensor):
        freq_weight = torch.exp(-self.factor_impu * freq_dict)
        weights = torch.sigmoid(torch.einsum("bij,jik->bjk",freq_weight.unsqueeze(-1),self.Dw)+\
                                torch.einsum("bij,ijk->bik", ht, self.W_d)+ self.b_d)
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
            prev_hidden, prev_cell,pre_we_decay, fgate_f = self.TLSTM_unit(prev_hidden,prev_cell, 
                                                                           inputs[i],times[i], 
                                                                           last_values[i],freqs[i])
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
                    time = second_times[i-1]
                second_hidden, second_cell,b_we_decay,fgate_b = self.TLSTM_unit(second_hidden, second_cell, 
                                                                                second_inputs[i], time,
                                                                                last_values[i],freqs[i])
                second_hidden_his += [second_hidden]
                second_weights_decay += [b_we_decay]
                second_weights_fgate += [fgate_b]
            second_hidden_his = torch.stack(second_hidden_his)
            second_weights_fgate = torch.stack(second_weights_fgate)
            second_weights_decay = torch.stack(second_weights_decay)
            weights_decay =torch.cat((weights_decay, second_weights_decay), dim=-1)
            weights_fgate =torch.cat((weights_fgate, second_weights_fgate), dim=-1)
            hidden_his = torch.cat((hidden_his, second_hidden_his), dim=-1)
            prev_hidden = torch.cat((prev_hidden, second_hidden), dim=-1)
            prev_cell = torch.cat((prev_cell, second_cell), dim=-1)
        if self.batch_first:
            hidden_his = hidden_his.permute(1, 0, 2, 3)
            weights_decay = weights_decay.permute(1, 0, 2, 3)
            weights_fgate = weights_fgate.permute(1, 0, 2, 3)
            
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", hidden_his, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*hidden_his, dim=1)
        hg = torch.cat([g_n, prev_hidden], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.max(betas, dim=1, keepdim=True).values
        mean = torch.sum(betas*mu, dim=1)
        return mean, alphas, betas , weights_decay, weights_fgate

class TimeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(TimeLSTM, self).__init__()
        # hidden dimensions
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_dim= output_dim
        # Temporal embedding AMITA LSTM
        self.amita_lstm = AMITA_LSTM(self.input_size , self.hidden_size, self.output_dim) 
    def forward(self,historic_features,timestamp, last_features, features_freqs , is_test=False):
        # Temporal features embedding
        outputs, alphas, betas, decay_weights, fgate = self.amita_lstm(historic_features,timestamp, 
                                                                    last_features, features_freqs)
        if is_test:
            return alphas, betas, decay_weights, fgate, outputs
        else:
            return outputs

class EarlyStopping:
    def __init__(self, mode, path, patience=3, delta=0):
        if mode not in {'min', 'max'}:
            raise ValueError("Argument mode must be one of 'min' or 'max'.")
        if patience <= 0:
            raise ValueError("Argument patience must be a positive integer.")
        if delta < 0:
            raise ValueError("Argument delta must not be a negative number.")

        self.mode = mode
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0

    def _is_improvement(self, val_score):
        """Return True iff val_score is better than self.best_score."""
        if self.mode == 'max' and val_score > self.best_score + self.delta:
            return True
        elif self.mode == 'min' and val_score < self.best_score - self.delta:
            return True
        return False

    def __call__(self, val_score, model):
        """
        Return True iff self.counter >= self.patience.
        """

        if self._is_improvement(val_score):
            self.best_score = val_score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print("Val loss improved, Saving model's best weights.")
            return False
        else:
            self.counter += 1
            print(f'Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                print(f'Stopped early. Best val loss: {self.best_score:.4f}')
                return True


class TrainerHelpers:
    def __init__(self, input_dim, hidden_dim, output_dim, device, optim, loss_criterion, schedulers, num_epochs, patience_n=50, task=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.optim = optim
        self.loss_criterion = loss_criterion
        self.schedulers = schedulers
        self.num_epochs = num_epochs
        self.patience_n = patience_n
        self.task = task

    @staticmethod
    def acc(predicted, label):
        predicted = predicted.sigmoid()
        pred = torch.round(predicted.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    def train_model(self, model, train_dataloader):
        modele.train()
        running_loss, running_corrects = 0.0, 0
        for bi, inputs in enumerate(tqdm(train_dataloader, total=len(train_dataloader), leave=False)):
            temporal_features, timestamp, last_data, data_freqs,labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            self.optim.zero_grad()
            _, _, _, _, outputs = model(temporal_features, timestamp,
                                        last_data, data_freqs, is_test=True)
            if self.task:
                loss = self.loss_criterion(outputs.sigmoid().squeeze(-1), labels.squeeze(-1))
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
                running_corrects += self.acc(outputs, labels)
            else:
                loss = self.loss_criterion(outputs.squeeze(-1), labels.squeeze(-1))
                loss.backward()
                self.optim.step()
                running_loss += loss.item()
        if self.task:
            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = running_corrects / len(train_dataloader.dataset)
            return epoch_loss, epoch_acc
        else:
            return running_loss / len(train_dataloader)

    def valid_model(self, model, valid_dataloader):
        model.eval()
        running_loss, running_corrects = 0.0, 0
        fin_targets, fin_outputs = [], []
        for bi, inputs in enumerate(tqdm(valid_dataloader, total=len(valid_dataloader), leave=False)):
            temporal_features, timestamp, last_data, data_freqs,labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            with torch.no_grad():
                _, _, _, _, outputs = model(temporal_features, timestamp,
                                            last_data, data_freqs, is_test=True)
            if self.task:
                loss = self.loss_criterion(outputs.sigmoid().squeeze(-1), labels.squeeze(-1))
                running_loss += loss.item()
                running_corrects += self.acc(outputs, labels)
            else:
                loss = self.loss_criterion(outputs.squeeze(-1), labels.squeeze(-1))
                running_loss += loss.item()
            fin_targets.append(labels.cpu().detach().numpy())
            fin_outputs.append(outputs.cpu().detach().numpy())
        if self.task:
            epoch_loss = running_loss / len(valid_dataloader)
            epoch_acc = running_corrects / len(valid_dataloader.dataset)
            return epoch_loss, epoch_acc, np.vstack(fin_targets), np.vstack(fin_outputs)
        else:
            mse = mean_squared_error(np.vstack(fin_targets), np.vstack(fin_outputs))
            mae = mean_absolute_error(np.vstack(fin_targets), np.vstack(fin_outputs))
            return running_loss / len(valid_dataloader), mse, mae, np.vstack(fin_targets), np.vstack(fin_outputs)

    def eval_model(self, model_class, model_path,  test_dataloader):
        # Initialize the model architecture
        model = model_class(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        # Load the model weights
        model.load_state_dict(torch.load(model_path))
        # Set the model to evaluation mode
        model.eval()
        fin_targets, fin_outputs = [], []
        all_alphas, all_betas, all_decays, fgate_weights = [], [], [], []
        for bi, inputs in enumerate(tqdm(test_dataloader, total=len(test_dataloader), leave=False, 
                                         desc='Evaluating on test data')):
            
            temporal_features, timestamp, last_data, data_freqs,labels = inputs
            temporal_features = temporal_features.to(torch.float32).to(self.device)
            timestamp = timestamp.to(torch.float32).to(self.device)
            last_data = last_data.to(torch.float32).to(self.device)
            data_freqs = data_freqs.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            with torch.no_grad():
                alphas, betas, decay_weights, fgate, outputs = model(temporal_features, timestamp,
                                                                     last_data, data_freqs, is_test=True)
            if self.task:
                fin_outputs.append(outputs.sigmoid().cpu().detach().numpy())
                
            else:
                fin_outputs.append(outputs.cpu().detach().numpy())
            fin_targets.append(labels.cpu().detach().numpy())
            all_alphas.append(alphas.cpu().detach().numpy())
            all_betas.append(betas.cpu().detach().numpy())
        return all_alphas, all_betas, all_decays, fgate_weights, np.vstack(fin_targets), np.vstack(fin_outputs)

    def train_validate_evaluate(self,model_class,  model, model_name, train_loader, val_loader, test_loader, params, model_path):
        best_losses, all_scores = [], []
        es = EarlyStopping(mode='min', path=f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                           patience=self.patience_n)
        for epoch in range(self.num_epochs):
            if self.task:
                loss, accuracy = self.train_model(model, train_loader)
                eval_loss, eval_accuracy, __, _ = self.valid_model(model, val_loader)
                if self.schedulers is not None:
                    self.schedulers.step()
                print(
                    f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, train loss: {loss:.8f}, accuracy: {accuracy:.8f} | valid loss: {eval_loss:.8f}, accuracy: {eval_accuracy:.4f}")
                if es(eval_loss, model):
                    best_losses.append(es.best_score)
                    print("best_score", es.best_score)
                    break
            else:
                loss = self.train_model(model, train_loader)
                eval_loss, mse_loss, mae_loss, _, _ = self.valid_model(model, val_loader)
                if self.schedulers is not None:
                    self.schedulers.step()
                print(
                    f"lr: {self.optim.param_groups[0]['lr']:.7f}, epoch: {epoch + 1}/{self.num_epochs}, train loss: {loss:.8f} | valid loss: {eval_loss:.8f} valid mse loss: {mse_loss:.8f}, valid mae loss: {mae_loss:.8f}")
                if es(mse_loss, model):
                    best_losses.append(es.best_score)
                    print("best_score", es.best_score)
                    break
        if self.task:
            _, _, y_true, y_pred = self.valid_model(model, val_loader)
            pr_score = average_precision_score(y_true, y_pred)
            print(f"[INFO] PR-AUC ON FOLD :{model_name} -  score val data: {pr_score:.4f}")
        else:
            _, _, _, y_true, y_pred = self.valid_model(modele, val_loader)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            print(
                f"[INFO] mse loss & mae loss on validation data Fold {model_name}: mse loss: {mse:.4f} - mae loss: {mae:.4f}")
        if self.task:
            f1_scores_folds = []
            targets, outputs = self._evaluate_model(model_class, f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                                                    test_loader)
            scores = self.metrics_binary(targets, outputs)
            delta, f1_scr = self.best_threshold(np.vstack(targets), np.vstack(outputs))
            f1_scores_folds.append((delta, f1_scr))
            all_scores.append((scores, f1_scores_folds))
            np.savez(os.path.join(model_path, f"results_data_{model_name}.npz"), 
                         auc_pr=scores, true_labels_data=np.vstack(outputs), 
                         predicted_labels_data= np.vstack(targets), 
                         folds_f1_scores= f1_scores_folds)
            print(f"[INFO] Results on test Folds {all_scores}")
        else:
            targets, outputs = self._evaluate_model(modele_class, f"{os.path.join(model_path, f'model_{model_name}.pth')}",
                                                    test_loader)
            scores = self.metrics_reg(targets, outputs, params)
            all_scores.append(scores)
            np.savez(os.path.join(model_path, f"test_data_fold_{model_name}.npz"), 
                      reg_scores=scores,true_labels=targets, predicted_labels= outputs)
            print(f"[INFO] Results on test Folds {all_scores}")
        return all_scores

    def _evaluate_model(self, model, model_path,  test_dataloader):
        targets, predicted = [], []
        all_alphas, all_betas, all_decays, fgate_weights = [], [], [], []
        alphas, betas, _, _, y_pred, y_true = self.eval_model(model, model_path, test_dataloader)
        targets.append(y_true)
        predicted.append(y_pred)
        all_alphas.append(alphas)
        all_betas.append(betas)
        targets_all = [np.vstack(targets[i]) for i in range(len(targets))]
        predicted_all = [np.vstack(predicted[i]) for i in range(len(predicted))]
        return targets_all, predicted_all

    @staticmethod
    def metrics_binary(targets, predicted):
        scores = []
        for y_true, y_pred, in zip(targets, predicted):
            fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_true)
            auc_score = metrics.auc(fpr, tpr)
            pr_score = metrics.average_precision_score(y_pred, y_true)
            scores.append([np.round(np.mean(auc_score), 4),
                           np.round(np.mean(pr_score), 4)])
        return scores

    @staticmethod
    def best_threshold(y_train,train_preds):
        delta, tmp = 0, [0, 0, 0]  # idx, cur, max
        for tmp[0] in tqdm(np.arange(0.1, 1.01, 0.01)):
            tmp[1] = f1_score(train_preds, np.array(y_train) > tmp[0])
            if tmp[1] > tmp[2]:
                delta = tmp[0]
                tmp[2] = tmp[1]
        print('best threshold is {:.2f} with F1 score: {:.4f}'.format(delta, tmp[2]))
        return delta, tmp[2]

    @staticmethod
    def adjusted_r2(actual: np.ndarray, predicted: np.ndarray, rowcount: np.int64, featurecount: np.int64):
        return 1 - (1 - r2_score(actual, predicted)) * (rowcount - 1) / (rowcount - featurecount)

    def metrics_reg(self, targets, predicted, rescale_params):
        scores = []
        for y_true, y_pred, in zip(targets, predicted):
            target_max, target_min = rescale_params['data_targets_max'], rescale_params['data_targets_min']
            targets_y_true = y_true * (target_max - target_min) + target_min
            targets_y_pred = y_pred * (target_max - target_min) + target_min
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            n = y_true.shape[0]
            r2 = r2_score(targets_y_true, targets_y_pred)
            adj_r2 = self.adjusted_r2(targets_y_true, targets_y_pred, n, self.input_dim)
            scores.append([rmse, mae, r2, adj_r2])
        return scores
