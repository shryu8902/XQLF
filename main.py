#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import LoadTimeSeriesDataSet
from torch.utils.data import DataLoader
from Model.MLP_mixer import MLP_MIXER
import pytorch_lightning as pl
from Model.MLP_mixer import MLP_MIXER2

#%%
LTSD_train = LoadTimeSeriesDataSet(target = 'USHOME',flag='train',scale=True, inverse = False, time_feature=True,freq='H')
LTSD_val = LoadTimeSeriesDataSet(target = 'USHOME',flag='val',scale=True, inverse = False, time_feature=True,freq='H')
LTSD_test = LoadTimeSeriesDataSet(target = 'USHOME',flag='test',scale=True, inverse = False, time_feature=True,freq='H')

loader_train = DataLoader(LTSD_train,batch_size=128, shuffle = True)
loader_val = DataLoader(LTSD_val,batch_size=128, shuffle = False)

model = MLP_MIXER(c_in = LTSD_train.c_in, d_model = 256, window = 168,
                n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=LTSD_train.time_feature)

model2 = MLP_MIXER2(c_in = LTSD_train.c_in, d_model = 256, window = 168,
                n_layers = 5, token_hdim = 128, ch_hdim = 128, dr_rates = 0.2, use_time_feature=LTSD_train.time_feature)

trainer = pl.Trainer()
trainer.fit(model2, loader_train)

# %%
