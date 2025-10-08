import torch
import copy
from torch import nn
import numpy as np
from cov import DilatedConvEncoder
from Enhancement import *

class BertInterpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * hidden_dim, input_dim)

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output

device = 'cuda'

class TSRL(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=1, dropout=0.2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.fc_reduce = torch.nn.Linear(output_dims, input_dims)

        self.feature_extractor = DilatedConvEncoder(
            # input_dims,
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.interphead = BertInterpHead(input_dims, output_dims)


        self.decoder_rnn_cell = nn.GRUCell(input_size=output_dims, hidden_size=hidden_dims)


        self.out = nn.Linear(hidden_dims, input_dims)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, X1, X2, X):  # X: B x T x input_dims
        # x1 = [torch.tensor(arr) for arr in x1]
        # x2 = [torch.tensor(arr) for arr in x2]
        m1 = (x1 != 0).to(torch.int)
        m2 = (x2 != 0).to(torch.int)
        # X1 = [torch.tensor(arr, dtype=torch.float32) for arr in X1]
        # X2 = [torch.tensor(arr, dtype=torch.float32) for arr in X2]

  
        if self.training:
            x_whole1 = self.input_fc(X1)
            x_whole1 = x_whole1.transpose(1, 2)
            x_whole1 = self.feature_extractor(x_whole1)  # B x Ch x T
            x_whole1 = x_whole1.transpose(1, 2)  # B x T x Co
            x_whole1 = self.repr_dropout(x_whole1)

            x_whole2 = self.input_fc(X2)
            x_whole2 = x_whole2.transpose(1, 2)
            x_whole2 = self.feature_extractor(x_whole2)  # B x Ch x T
            x_whole2 = x_whole2.transpose(1, 2)  # B x T x Co
            x_whole2 = self.repr_dropout(x_whole2)


        if self.training:
            x_interp1 = self.input_fc(x1)
            x_interp1 = x_interp1.transpose(1, 2)
            x_interp1 = self.feature_extractor(x_interp1)  # B x Ch x T
            x_interp1 = x_interp1.transpose(1, 2)  # B x T x Co
            x_interp1 = self.repr_dropout(x_interp1)
            x_interp1 = self.interphead(x_interp1)

            x_interp2 = self.input_fc(x2)
            x_interp2 = x_interp2.transpose(1, 2)
            x_interp2 = self.feature_extractor(x_interp2)  # B x Ch x T
            x_interp2 = x_interp2.transpose(1, 2)  # B x T x Co
            x_interp2 = self.repr_dropout(x_interp2)
            x_interp2 = self.interphead(x_interp2)

            mask1 = (~torch.isnan(x_interp1).any(dim=-1)).float()  # (sample_num, seq_len)
            # (sample_num, seq_len)
            mask2 = (~torch.isnan(x_interp2).any(dim=-1)).float()  # (sample_num, seq_len)

            sample_num, seq_len, _ = x_interp1.shape
            device = x_interp1.device


            hidden_state1 = torch.zeros(sample_num, self.decoder_rnn_cell.hidden_size, device=device)
            hidden_state2 = torch.zeros(sample_num, self.decoder_rnn_cell.hidden_size, device=device)


            outputs1 = []
            outputs2 = []


            self.fc = torch.nn.Linear(self.input_dims, self.output_dims).to(device)
            self.out = torch.nn.Linear(self.decoder_rnn_cell.hidden_size, self.output_dims).to(device)


            for t in range(seq_len):
                y_t1 = x_interp1[:, t, :]
                y_t2 = x_interp2[:, t, :]


                y_t1 = self.fc(y_t1)  # (sample_num, 320)
                y_t2 = self.fc(y_t2)  # (sample_num, 320)


                hidden_state1 = self.decoder_rnn_cell(y_t1, hidden_state1)
                hidden_state2 = self.decoder_rnn_cell(y_t2, hidden_state2)


                output_t1 = self.out(hidden_state1)  # (sample_num, 320)
                output_t2 = self.out(hidden_state2)  # (sample_num, 320)


                output_t1 = torch.where(mask1[:, t].unsqueeze(-1) == 1, y_t1, output_t1)
                output_t2 = torch.where(mask1[:, t].unsqueeze(-1) == 1, y_t1, output_t2)


                outputs1.append(output_t1)
                outputs2.append(output_t2)


            outputs1 = torch.stack(outputs1, dim=1)
            outputs2 = torch.stack(outputs2, dim=1)
            outputs2 = outputs2.clone().detach()
            outputs2 = self.fc_reduce(outputs2)
            outputs1 = outputs1.clone().detach()
            outputs1 = self.fc_reduce(outputs1)


        # x = [torch.tensor(arr) for arr in X]
        X = X.float()
        x = self.input_fc(X)
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)  # B x Ch x T
        x = x.transpose(1, 2)  # B x T x Co
        x = self.repr_dropout(x)

        if self.training:
            return x_whole1, x_whole2, outputs1, outputs2, m1, m2, x1, x2
        else:
            return x


