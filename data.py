#%%
from ast import parse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
#%%
class LTS_dataprep():
    def __init__(self, target, json_root='./Data/data_info.json', flag='train', scale=True, inverse=False, time_feature = True, freq='H'):
        assert target in ['ECL', 'ELIA', 'KAGGLE', 'PANAMA','PRECON','UCI','UMASS','US10','USHOME']
        assert flag in ['train','val','test']
        type_map = {'train':0, 'val':1, 'test':2}
        day_len_map = {'H':24, 'Q':96, 'M': 1440}

        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.set_type = type_map[flag]
        self.day_len = day_len_map[freq]
        with open(json_root, 'r') as j:
            contents = json.loads(j.read())
        self.meta = contents[target]
        self.add_var = True if len(self.meta['add_cols'])>0 else False

    def __read_data__(self):
        df_raw = pd.read_csv(meta['path'],parse_dates = [meta['date_col']],index_col=[meta['date_col']])
        assert len(df_raw)%self.day_len==0
        self.n_dates = len(df_raw)//self.day_len
        split_indx = [0, np.int(self.n_dates*0.6)*self.day_len, np.int(self.n_dates*0.8)*self.day_len, self.n_dates*self.day_len]

        split_indx =[0, np.int(n_dates*0.6)*24, np.int(n_dates*0.8)*24, len(df_raw)]

        df_train = df_raw.iloc[split_indx[0]:split_indx[1]]
        df_target = df_raw.iloc[split_indx[set_type]:split_indx[set_type+1]] #self.set_type

        if self.scale:
            self.load_scaler = StandardScaler()
            self.add_scaler = StandardScaler()
            self.load_scaler.fit(df_train[meta["load_col"]].values.reshape(-1,1)) #self.meta

            load = load_scaler.transform(df_target[meta["load_col"]].values.reshape(-1,1)) #self.meta, self.load_scaler

            if self.add_var :
                self.add_scaler.fit(df_train[meta["add_cols"]].values) #self.meta
                add_vars = add_scaler.transform(df_target[meta["add_cols"]].values) #self.meta, self.add_scaler
        else :
            load = df_target[meta["load_col"]].values.reshape(-1,1)) #self.meta, self.load_scaler
            if self.add_var:
                add_vars = df_target[meta["add_cols"]].values
    
        if time_feature:
            time_series = pd.Series(df_target.index)
            hour = time_series.dt.hour
            month = time_series.dt.month
            weekday = time_series.dt.weekday #mon~sun 1~7
            week = time_series.dt.week #i-th week 0~52
            season = time_series.dt.month%12 // 3 + 1
            day = time_series.dt.dayofyear # 0~365
            time_features = np.concatenate([hour,month,weekday,week,season,day],axis=1)
        
            X = np.concatenate([load, time_features, add_vars],axis=1) if self.add_var else load
        else:
            X = np.concatenate([load, add_vars],axis=1) if self.add_var else load
            
        self.X = X




