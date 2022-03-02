#!/usr/bin/env python
# coding: utf-8

# ## Configs

# In[1]:


offset = 0
limit = 3000
period = 'max' # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max


# ## Download all NASDAQ traded symbols

# In[2]:


import pandas as pd

data = pd.read_csv("http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
data_clean = data[data['Test Issue'] == 'N']
symbols = data_clean['NASDAQ Symbol'].tolist()
print('total number of symbols traded = {}'.format(len(symbols)))


# ## Download Historic data

# In[3]:


get_ipython().system(' pip install yfinance > /dev/null 2>&1')
get_ipython().system(' mkdir hist')


# In[4]:


import yfinance as yf
import os, contextlib


# In[5]:


get_ipython().run_cell_magic('time', '', "\nlimit = limit if limit else len(symbols)\nend = min(offset + limit, len(symbols))\nis_valid = [False] * len(symbols)\n# force silencing of verbose API\nwith open(os.devnull, 'w') as devnull:\n    with contextlib.redirect_stdout(devnull):\n        for i in range(offset, end):\n            s = symbols[i]\n            data = yf.download(s, period=period)\n            if len(data.index) == 0:\n                continue\n        \n            is_valid[i] = True\n            data.to_csv('hist/{}.csv'.format(s))\n\nprint('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))")


# In[6]:


valid_data = data_clean[is_valid]
valid_data.to_csv('symbols_valid_meta.csv', index=False)


# ## Separating ETFs and Stocks

# In[7]:


get_ipython().system('mkdir stocks')
get_ipython().system('mkdir etfs')


# In[8]:


etfs = valid_data[valid_data['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
stocks = valid_data[valid_data['ETF'] == 'N']['NASDAQ Symbol'].tolist()


# In[9]:


import shutil
from os.path import isfile, join

def move_symbols(symbols, dest):
    for s in symbols:
        filename = '{}.csv'.format(s)
        shutil.move(join('hist', filename), join(dest, filename))
        
move_symbols(etfs, "etfs")
move_symbols(stocks, "stocks")


# In[10]:


get_ipython().system(' rmdir hist')

