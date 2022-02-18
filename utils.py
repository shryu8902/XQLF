import numpy as np
import pandas as pd
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def time_encode(time_data):
    time_series = pd.Series(time_data)   #time_series = pd.Series(df_target.index)
    hour = time_series.dt.hour
    month = time_series.dt.month
    weekday = time_series.dt.weekday #mon~sun 1~7
    week = time_series.dt.week #i-th week 0~52
    quarter = time_series.dt.quarter #season 1~4
    day = time_series.dt.dayofyear # 0~365