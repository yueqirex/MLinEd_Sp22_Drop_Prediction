import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import pandas as pd
np.random.seed(42)


def train_val_test_split(group_all_user):
    uid_pool = group_all_user.index.values.copy()
    idx_pool = np.arange(uid_pool.shape[0])
    np.random.shuffle(idx_pool)
    train_idx, val_idx, test_idx =\
        idx_pool[:int(0.8*uid_pool.shape[0])],\
        idx_pool[int(0.8*uid_pool.shape[0]):int(0.9*uid_pool.shape[0])],\
        idx_pool[int(0.9*uid_pool.shape[0]):]
    train_uid = uid_pool[train_idx]
    val_uid = uid_pool[val_idx]
    test_uid = uid_pool[test_idx]
    return train_uid, val_uid, test_uid


def batch_gene_input_data(group_input, group_label, batch_uid, padding_mask):
    batch_cid = []
    batch_action = []
    batch_cid_label = []
    batch_action_label = []
    max_seq_len = 0
    max_index_len = 0
    # generate batch ids
    for uid in batch_uid:
        # input
        cid, action = group_input[uid][0], group_input[uid][1]
        if cid.shape[0] > max_seq_len:
            max_seq_len = cid.shape[0]
        batch_cid.append(cid.copy().tolist())
        batch_action.append(action.copy().tolist())
        # label
        label = group_label[uid]
        cid_label, action_label = label[0], label[1]
        if cid_label.shape[0] > max_index_len:
            max_index_len = cid_label.shape[0]
        batch_cid_label.append(cid_label.copy().tolist())
        batch_action_label.append(action_label.copy().tolist())
    # pad batch data
    for i in range(len(batch_cid)):
        if len(batch_cid[i]) < max_seq_len:
            batch_cid[i] = batch_cid[i] + (max_seq_len - len(batch_cid[i])) * [padding_mask] # mask=n_crs
            batch_action[i] = batch_action[i] + (max_seq_len - len(batch_action[i])) * [2] # padding token other than 0,1
        if len(batch_cid_label[i]) < max_index_len:
            batch_cid_label[i] = batch_cid_label[i] + (max_index_len - len(batch_cid_label[i])) * [padding_mask]
            batch_action_label[i] = batch_action_label[i] + (max_index_len - len(batch_action_label[i])) * [padding_mask]
    return batch_cid, batch_action, batch_cid_label, batch_action_label

def get_global_max_seq_len(group_input):
    max_seq_len = 0
    all_uid = group_input.index.values.copy()
    for uid in all_uid:
        cid, action = group_input[uid][0], group_input[uid][1]
        if cid.shape[0] > max_seq_len:
            max_seq_len = cid.shape[0]
    return max_seq_len

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4).to(device), requires_grad=True)
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4).to(device), requires_grad=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4).to(device), requires_grad=True)
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(self.device), 
                        torch.zeros(bs, self.hidden_size).to(self.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class DropPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_courses, num_actions, device):
        super(DropPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.num_courses = num_courses,
        self.num_actions = num_actions
        
        self.course_emb = nn.Parameter(torch.FloatTensor(num_courses+1, hidden_size).to(device), requires_grad=True) # for padding
        self.action_emb = nn.Parameter(torch.FloatTensor(num_actions+1, hidden_size).to(device), requires_grad=True) 

        self.lstm = CustomLSTM(2*input_size, hidden_size, device = device)
        self.out_layer = nn.Linear(hidden_size, num_courses+1) 
        self.reset_parameters()
    
    def reset_parameters(self):
        initrange = 1 / math.sqrt(self.hidden_size / 2)
        self.course_emb.data.normal_(0, initrange)
        self.action_emb.data.normal_(0, initrange)
        self.out_layer.bias.data.fill_(0.1)
        self.out_layer.weight.data.normal_(0, initrange)
    
    def forward(self, batch_cid, batch_action):
        batch_cid = torch.tensor(batch_cid).to(self.device)
        batch_action = torch.tensor(batch_action).to(self.device)
        # print(batch_cid.shape)
        assert batch_action.shape == batch_cid.shape
        batch_size, seq_len = batch_cid.shape # (bs, seq_len)
        batch_cid_1d = batch_cid.clone().view(-1)
        batch_action_1d = batch_action.clone().view(-1)

        batch_cid_emb = self.course_emb[batch_cid_1d].view(batch_size, seq_len, -1)
        batch_action_emb = self.action_emb[batch_action_1d].view(batch_size, seq_len, -1)

        batch_concat_emb = torch.concat((batch_cid_emb, batch_action_emb), dim=-1)
        hidden_state, _ = self.lstm(batch_concat_emb) # (bs, seq_len, d_)
        last_hidden = hidden_state[:,-1,:] # (bs, d_) -> logits
        output = self.out_layer(last_hidden)
        return output

    def loss(self, output, batch_cid_label, batch_action_label, padding_mask):
        batch_cid_label = torch.tensor(batch_cid_label).to(self.device)
        batch_action_label = torch.tensor(batch_action_label).to(self.device)
        tar_output = torch.gather(output, -1, batch_cid_label) # (bs, d_padded)
        valid_pos = (batch_cid_label != padding_mask).nonzero(as_tuple=True)
        tar_out = tar_output[valid_pos]
        tar_label = batch_action_label[valid_pos]
        bce_loss_logits = nn.BCEWithLogitsLoss()
        loss = bce_loss_logits(tar_out, tar_label.float()) # (bs, d_model)
        return loss, tar_out.view(-1), tar_label.view(-1)
    
    def get_course_emb(self, semester_cid_index):
        return self.course_emb[semester_cid_index].clone().data.cpu().numpy()


class LinearPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_courses, num_actions, global_max_seq_len, padding_mask, device):
        super(LinearPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.num_courses = num_courses,
        self.num_actions = num_actions
        self.max_len = global_max_seq_len
        self.padding_mask = padding_mask
        
        self.course_emb = nn.Parameter(torch.FloatTensor(num_courses+1, hidden_size).to(device), requires_grad=True) # for padding
        self.action_emb = nn.Parameter(torch.FloatTensor(num_actions+1, hidden_size).to(device), requires_grad=True) 

        self.linear_layer = nn.Linear(self.max_len*2*input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, num_courses+1) 
        self.reset_parameters()
    
    def reset_parameters(self):
        initrange = 1 / math.sqrt(self.hidden_size / 2)
        self.course_emb.data.normal_(0, initrange)
        self.action_emb.data.normal_(0, initrange)
        self.linear_layer.bias.data.fill_(0.1)
        self.linear_layer.weight.data.normal_(0, initrange)
        self.out_layer.bias.data.fill_(0.1)
        self.out_layer.weight.data.normal_(0, initrange)
    
    def forward(self, batch_cid, batch_action):
        batch_cid = torch.tensor(batch_cid).to(self.device)
        batch_action = torch.tensor(batch_action).to(self.device)
        # print(batch_cid.shape)
        assert batch_action.shape == batch_cid.shape
        batch_size, seq_len = batch_cid.shape # (bs, seq_len)
        batch_cid_1d = batch_cid.clone().view(-1)
        batch_action_1d = batch_action.clone().view(-1)

        batch_cid_emb = self.course_emb[batch_cid_1d].view(batch_size, seq_len, -1)
        batch_action_emb = self.action_emb[batch_action_1d].view(batch_size, seq_len, -1)

        batch_concat_emb = torch.concat((batch_cid_emb, batch_action_emb), dim=-1)
        batch_concat_emb = batch_concat_emb.view(batch_size, -1) # (bs, seq_len*d_model)
        if batch_concat_emb.shape[-1] < self.max_len*2*self.input_size:
            padding_part = torch.full(size=(batch_concat_emb.shape[0], self.max_len*2*self.input_size - batch_concat_emb.shape[-1]),
                                      fill_value=self.padding_mask).to(batch_concat_emb.device)
            batch_concat_emb = torch.concat((batch_concat_emb, padding_part), dim=-1)
        last_hidden = self.linear_layer(batch_concat_emb)
        output = self.out_layer(last_hidden)
        return output

    def loss(self, output, batch_cid_label, batch_action_label, padding_mask):
        batch_cid_label = torch.tensor(batch_cid_label).to(self.device)
        batch_action_label = torch.tensor(batch_action_label).to(self.device)
        tar_output = torch.gather(output, -1, batch_cid_label) # (bs, d_padded)
        valid_pos = (batch_cid_label != padding_mask).nonzero(as_tuple=True)
        tar_out = tar_output[valid_pos]
        tar_label = batch_action_label[valid_pos]
        bce_loss_logits = nn.BCEWithLogitsLoss()
        loss = bce_loss_logits(tar_out, tar_label.float()) # (bs, d_model)
        return loss, tar_out.view(-1), tar_label.view(-1)
    
    def get_course_emb(self, semester_cid_index):
        return self.course_emb[semester_cid_index].clone().data.cpu().numpy()