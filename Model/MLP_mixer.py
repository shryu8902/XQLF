#%%
import numpy as np
import pandas as pd
from torch.nn.modules.dropout import Dropout
pd.options.display.float_format = '{:,.2f}'.format 

#torch related
import torch
from torch import nn
import torch.nn.functional as F

#pytorch lightning
import pytorch_lightning as pl

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from Model.embed import DataEmbedding
from Model.common_model import LTS_model

# %%
class MLP(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class MixerLayer(nn.Module):
    def __init__(self, token_dim, ch_dim, token_hdim = 256, ch_hdim = 256):
        super().__init__()
        self.token_dim = token_dim
        self.ch_dim = ch_dim
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim

        self.token_mix = nn.Sequential(
            nn.LayerNorm(ch_dim),
            Rearrange('b n d -> b d n'), # (batch, length, depth) -> (batch, depth, length)
            MLP(token_dim, hidden_size=token_hdim),#             output: (batch, depth, length)
            Rearrange('b d n -> b n d')  # (batch, depth, length) -> (batch, length, depth)
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(ch_dim),
            MLP(ch_dim, hidden_size=ch_hdim)    # output : (batch, length, depth)
        )
        
    def forward(self, x): 
        x = x + self.token_mix(x) 
        x = x + self.channel_mix(x)
        
        return x


# %%
class MLP_MIXER(LTS_model):
    def __init__(self, c_in, d_model, window = 168, n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=True):
        super().__init__()
        self.d_model = d_model
        self.c_in = c_in # number of variables except time.
        self.window = window
        self.token_hdim = token_hdim
        self.ch_hdim = ch_hdim
        self.n_layers=n_layers
        self.dr_rates = dr_rates
        self.use_time_feature = use_time_feature
        self.mixer_layers = nn.ModuleList([])
        self.input_embedding = DataEmbedding(c_in, d_model, use_time_feature=self.use_time_feature)

        for _ in range(n_layers):
            self.mixer_layers.append(
                MixerLayer(token_dim  = self.window, ch_dim=self.d_model, token_hdim=self.token_hdim, ch_hdim=self.ch_hdim)
            )
            self.mixer_layers.append(
            nn.Dropout(p=self.dr_rates)
            )
        self.mixer_layers.append(
                Rearrange('b n d -> b d n')
            )
        self.mixer_layers.append(
            nn.AvgPool1d(kernel_size=self.window)
        )
        self.mlp_head = nn.Linear(d_model, 24)

    def forward(self, x):
        x = self.input_embedding(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        x = x.flatten(start_dim=1)
        out = self.mlp_head(x)
        y_pred=out
        return y_pred
