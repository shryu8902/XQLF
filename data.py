#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json

from torch.utils.data import Dataset

#%%
class LoadTimeSeriesDataSet(Dataset):
    '''
    target : name of datasets
    scale : use standard scaler or not
    inverse : ?
    time_feature : convert time index as features
    set_type : train/validation/test
    day_len : length of day by data frequency
    meta : meta data for target dataset
    add_var : True if additional data is available
    n_dates : total number of dates in target dataset
    n_dates_set : number of dates in current dataset (train/val/test)

    X : TS formed current dataset (train/val/test)
    load_scaler : scaler for load data
    add_scaler : scaler for additional variables 
    '''    
    def __init__(self, target, json_root='./Data/data_info.json', flag='train', scale=True, inverse=False, time_feature = True, freq='H'):
        assert target in ['ECL', 'ELIA', 'KAGGLE', 'PANAMA','PRECON','UCI','UMASS','US10','USHOME']
        assert flag in ['train','val','test']
        type_map = {'train':0, 'val':1, 'test':2}
        day_len_map = {'H':24, 'Q':96, 'M': 1440}
        self.past_window=7
        self.target = target
        self.scale = scale
        self.freq=freq
        self.inverse = inverse
        self.time_feature = time_feature
        self.set_type = type_map[flag]
        self.day_len = day_len_map[freq]
        with open(json_root, 'r') as j:
            contents = json.loads(j.read())
        self.meta = contents[target]
        self.add_var = True if len(self.meta['add_cols'])>0 else False
        self.unit = self.meta["unit"]
        self.__read_data__()
        self.c_in = int(len(self.meta['add_cols'])+1)

    def __read_data__(self):
        df_raw = pd.read_csv(self.meta['path'],parse_dates = [self.meta['date_col']],index_col=[self.meta['date_col']])
        # df_raw = pd.read_csv(meta['path'],parse_dates = [meta['date_col']],index_col=[meta['date_col']])
        assert len(df_raw)%self.day_len==0, 'dataset length error'
        self.n_dates = len(df_raw)//self.day_len
        # n_dates = len(df_raw)//day_len
        split_indx = [0, np.int(self.n_dates*0.6)*self.day_len, np.int(self.n_dates*0.8)*self.day_len, self.n_dates*self.day_len]
        # split_indx = [0, np.int(n_dates*0.6)*day_len, np.int(n_dates*0.8)*day_len, n_dates*day_len]
            
        df_train = df_raw.iloc[split_indx[0]:split_indx[1]]
        df_target = df_raw.iloc[split_indx[self.set_type]:split_indx[self.set_type+1]] #self.set_type
        # df_target = df_raw.iloc[split_indx[set_type]:split_indx[set_type+1]] #self.set_type

        if self.scale:
            self.load_scaler = StandardScaler()
            self.add_scaler = StandardScaler()
            self.load_scaler.fit(df_train[self.meta["load_col"]].values.reshape(-1,1)) #self.meta
            # load_scaler.fit(df_train[meta["load_col"]].values.reshape(-1,1))
            load = self.load_scaler.transform(df_target[self.meta["load_col"]].values.reshape(-1,1)) #self.meta, self.load_scaler
            # load = load_scaler.transform(df_target[meta["load_col"]].values.reshape(-1,1)) #self.meta, self.load_scaler

            if self.add_var :
                self.add_scaler.fit(df_train[self.meta["add_cols"]].values) 
                # add_scaler.fit(df_train[meta["add_cols"]].values) #self.meta
                add_vars = self.add_scaler.transform(df_target[self.meta["add_cols"]].values) #self.meta, self.add_scaler
                # add_vars = add_scaler.transform(df_target[meta["add_cols"]].values) #self.meta, self.add_scaler
        else :
            load = df_target[self.meta["load_col"]].values.reshape(-1,1) #self.meta, self.load_scaler
            if self.add_var:
                add_vars = df_target[self.meta["add_cols"]].values
    
        if self.time_feature:
            time_series = pd.Series(df_target.index)            
            hour = time_series.dt.hour  # 0~23
            month = time_series.dt.month # 1~12
            weekday = time_series.dt.weekday #mon~sun 0~6
            week = time_series.dt.week #i-th week 1~53
            season = time_series.dt.month%12 // 3 + 1 # 1~4
            day = time_series.dt.dayofyear # 1~366
            time_features = np.stack([hour,month,weekday,week,season,day],axis=1)        

            X = np.concatenate([time_features, load, add_vars],axis=1) if self.add_var else load
        else:
            X = np.concatenate([load, add_vars],axis=1) if self.add_var else load
            
        self.L, self.D = X.shape
        self.n_dates_set = len(X)//self.day_len
        self.X = X.reshape((-1,self.day_len,self.D)).astype('float32')

    def __len__(self):
        return self.n_dates_set-self.past_window
        #return self.n_datses_set-self.past_window - self.future_window + 1 if forecast more than 1 day
    
    def __getitem__(self, index):
        input_begin = index
        input_end = input_begin + self.past_window

        seq_x = self.X[input_begin:input_end,...].reshape(-1, self.D)
        # seq_y = self.X[input_end,...].reshape(-1, self.D)
        if self.time_feature:
            seq_y = self.X[input_end,:,6]
        else:
            seq_y = self.X[input_end,:,0]
        if self.inverse:
            seq_y = self.load_scaler.inverse_transform(seq_y.reshape(-1,self.day_len)).ravel() 
        return seq_x, seq_y # seqx : [b, w, d], seq_y : [b, 24]




