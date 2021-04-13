# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dự đoán giá chứng khoán trong 1 ngày tiếp theo của Amazon
# ---

# %% [markdown]
# ## 1. Import thư viện
# ---

# %%
import pandas as pd 

# phải cài cái này để load dữ liệu về


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.linear_model import LogisticRegression
# vẽ biểu đồ cho ngầu 
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# %matplotlib inline

# xử lý thời gian 
from datetime import datetime

# keras for LSTM
# phải cài cái này 
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM

# %% [markdown]
# ## 2. READ DATA
# ---

# %%
df = pd.read_csv('BTCUSDT_15MinuteBars.csv')

df
# df = pd.read_csv('AMZN.csv')
# df = df.set_index('Date')
# df

# %% [markdown]
# cộng , trừ => nến

# %%
def Create_Train(df,n,percent):

        df.drop('timestamp', inplace = True, axis = 1 )
        df.drop('close_time', inplace = True, axis = 1 )
        df.drop('quote_av', inplace = True, axis = 1 )
        df.drop('trades', inplace = True, axis = 1 )
        df.drop('tb_base_av', inplace = True, axis = 1 )
        df.drop('tb_quote_av', inplace = True, axis = 1 )
        df.drop('ignore', inplace = True, axis = 1 )
        df.drop('volume', inplace = True, axis = 1 )
        #df.drop('open',inplace = True, axis = 1)

        blocks = list()
        return_Df= df[:-n]
        return_Df.reset_index(inplace= True)
        return_Df.drop('index',inplace = True, axis = 1)
        for i in range(1,n):
            blocks.append(df[i:-(n-i)])
            blocks[i-1].reset_index(inplace= True)
            blocks[i-1].drop('index',inplace = True, axis = 1)
            blocks[i-1].rename(columns={"open":"open " + str(i),"high": "high "+ str(i), "low": "low "+ str(i), "close": "close "+ str(i)},inplace = True)
            return_Df = return_Df.join(blocks[i-1])
        return_Df = return_Df.drop(return_Df.tail(1).index)#?

        view =df['close'][n+1:]

        view.reset_index(inplace=True, drop=True)
        #index = ()
        #print(view > return_Df['close'])
        #print(view.index.difference(index))
        index1 = view > (return_Df['close '+str(n-1)]*percent)

        index2 = view <= (return_Df['close '+str(n-1)]*percent)

        view[index1]='Up' 
        view[index2]='Down' 
        view.rename("Tag",inplace = True)

        return_Df = return_Df.join(view)


        return return_Df
b=  Create_Train(df,15,1.0005)  

# %%
b.isna().sum()


# %%
def Create_Train_candel(df,n):# high - open, open - close, close - low 
    return_df = pd.concat([(df["high"]- df["open"]).rename("high"),(df["open"]- df["close"]).rename("now"),(df["close"] - df["low"]).rename("low")],axis=1)
    for i in range(1,n):
        return_df = return_df.join((df["high "+ str(i)]- df["open "+str(i)]).rename("high "+str(i)))
        return_df = return_df.join((df["open "+ str(i)] - df["close " +str(i)]).rename("now "+str(i)))
        return_df = return_df.join((df["close "+ str(i)] - df["low "+str(i)]).rename("low "+str(i)))
    return return_df.join(b["Tag"])
test =Create_Train_candel(b,15)


# %%
def Percent_df(df,n):
        # df.drop('timestamp', inplace = True, axis = 1 )
        # df.drop('close_time', inplace = True, axis = 1 )
        # df.drop('quote_av', inplace = True, axis = 1 )
        # df.drop('trades', inplace = True, axis = 1 )
        # df.drop('tb_base_av', inplace = True, axis = 1 )
        # df.drop('tb_quote_av', inplace = True, axis = 1 )
        # df.drop('ignore', inplace = True, axis = 1 )
        # df.drop('volume', inplace = True, axis = 1 )
        #df.drop('open',inplace = True, axis = 1)

        blocks = list()
        return_Df= df[:-n]
        return_Df.reset_index(inplace= True)
        return_Df.drop('index',inplace = True, axis = 1)
        for i in range(1,n):
            blocks.append(df[i:-(n-i)])
            blocks[i-1].reset_index(inplace= True)
            blocks[i-1].drop('index',inplace = True, axis = 1)
            blocks[i-1].rename(columns={"open":"open " + str(i),"high": "high "+ str(i), "low": "low "+ str(i), "close": "close "+ str(i)},inplace = True)
            return_Df = return_Df.join(blocks[i-1])
        return_Df.drop(return_Df.tail(1).index,inplace=True)

        view =df['close'][n+1:]

        view.reset_index(inplace=True, drop=True)
        #index = ()
        #print(view > return_Df['close'])
        #print(view.index.difference(index))
        view = (view /(return_Df['close 7'])-1)*100

        # view[index1]='Up' 
        # view[index2]='Down' 
        view.rename("Tag",inplace = True)

        return_Df = return_Df.join(view)


        return return_Df
c=  Percent_df(df,8)  

# %%
test.isna().sum()

# %%
test

# %%
b= test

# %%

# %%
b

# %%
b['Tag'].value_counts()

# %% [markdown]
# index add weight
#

# %%
#arr=
#b = b.drop(b[b.max(axis=1)==0].index)

#Tag = b['Tag']

#CHIA MAX SCALER
#arr = b.drop('Tag',axis=1).div(b.max(axis=1),axis=0)

#arr = arr.join(Tag)
arr=b

# %%
arr

# %%

# %%
(b.max(axis=1)==0).sum()

# %%
b.drop('Tag',axis=1).div(b.max(axis=1),axis=0).isna().sum()

# %%

# %%
y_sr = arr["Tag"] 
X_df = arr.drop("Tag", axis=1)

# %%
train_X_df.isna().sum()

# %%

train_X_df, val_X_df, train_y_sr, val_y_sr = train_test_split(X_df, y_sr, test_size=0.3, random_state=0)

# %%

# %%
mlpc = MLPClassifier(hidden_layer_sizes=(100),random_state=0, max_iter=2500)

# %%
full_pipeline = make_pipeline(MinMaxScaler(),mlpc)


# %%
full_pipeline.fit(train_X_df, train_y_sr)

# %%
full_pipeline.score(val_X_df , val_y_sr )

# %%
full_pipeline.fit(val_X_df , val_y_sr )

# %%
full_pipeline.score(train_X_df, train_y_sr)

# %% [markdown]
# # Làm quả hàm chuẩn bị dữ lịu tét

# %%
b

# %%
full_pipeline.predict(val_X_df.tail(10))

# %%
full_pipeline.predict_proba(val_X_df.tail(10))

# %%
val_y_sr.tail(10)

# %%
pro_arr = full_pipeline.predict_proba(val_X_df)

# %%
np.sum((pro_arr[:,1]>0.6))

# %%
from binance.client import Client
import datetime

# YOUR API KEYS HERE
#api_key = "1xkfS28jR7eW8EZk77BCt3NtznzgIoJXjxTelowoDPD8MXvBUfBazJV0CSzF0Jwq"
#api_secret = "iTqdOQHe32KQvhG33yC68bk5wTKRkb2zlszwM9x63sXyMjwxAYe6WerSj5SW4g5l"

api_key = "ANUulMmh5ClMUu4lFgoahqBoqgIFCPGAlz2vIwHdjk80HQSMp9miHKwECPdiCAVJ"    #Enter your own API-key here
api_secret = "mv8RXmGzA4dLC5wgXvb4Op14otLUNm3uEU9egCxRYX2h3S2Ns7mrnHifg7Hu8gQS" #Enter your own API-secret here
bclient = Client(api_key=api_key, api_secret=api_secret)

start_date = datetime.datetime.strptime('23 May 2019', '%d %b %Y')
today = datetime.datetime.today()

def binanceBarExtractor(symbol):
    print('working...')
    filename = '{}_100MinuteBars.csv'.format(symbol)

    klines = bclient.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, start_date.strftime("%d %b %Y %H:%M:%S"), today.strftime("%d %b %Y %H:%M:%S"), 1000)
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    data.set_index('timestamp', inplace=True)
    data.to_csv(filename)
    print('finished!')


if __name__ == '__main__':
    # Obviously replace BTCUSDT with whichever symbol you want from binance
    # Wherever you've saved this code is the same directory you will find the resulting CSV file
    #binanceBarExtractor('BTCUSDT')
    balance = bclient.get_asset_balance(asset='REEF')
    print(balance)


# %%
df2 = pd.read_csv('BTCUSDT_30MinuteBars_by_Now.csv ')
df2

# %%
data = Create_Train(df2,8)

# %%
data.tail(1)

# %%
full_pipeline.predict(data.tail(1).drop('Tag',axis=1))

# %%
