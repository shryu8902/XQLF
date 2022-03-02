from calendar import month_name
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SinusoidalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(SinusoidalEncoding, self).__init__()

        w = torch.zeros(max_len, d_model).float()
        w.require_grad = False

        position = torch.arange(0,max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb=nn.Embedding.from_pretrained(w, freeze=True)

    def forward(self, x):
        return self.emb(x).detach()

class TimeFeatureEncoding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEncoding, self).__init__()
        self.time_enc =SinusoidalEncoding(d_model, max_len=500)  
    
    def forward(self, x):
        x = x.int()
        hour_enc = self.time_enc(x[...,0])
        month_enc = self.time_enc(x[...,1])
        weekday_enc = self.time_enc(x[...,2])
        week_enc = self.time_enc(x[...,3])
        season_enc = self.time_enc(x[...,4])
        day_enc = self.time_enc(x[...,5])

        return hour_enc + month_enc + weekday_enc + week_enc + season_enc + day_enc

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, use_time_feature = True):
        super(DataEmbedding, self).__init__()
        self.use_time_feature = use_time_feature
        self.value_embedding = nn.Linear(c_in, d_model)
        self.positional_embedding = PositionalEncoding(d_model=d_model)
        if self.use_time_feature:
            self.time_embedding = TimeFeatureEncoding(d_model=d_model)

    def forward(self, x):
        if self.use_time_feature:
            x_embed = self.value_embedding(x[...,6:]) + self.positional_embedding(x) + self.time_embedding(x[...,:6])
        else :
            x_embed = self.value_embedding(x) + self.positional_embedding(x)
        
        return x_embed