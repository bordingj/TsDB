


import pandas as pd
import numpy as np
import os
from tsdb.tsframe import TsFrame
    
class TickDB(object):
    
    def __init__(self, 
                 rootdir, 
                 chunklen=131072):
                     
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        if rootdir[-1] == '/' or rootdir[-1] == '\\':
            rootdir = rootdir[:-1]
        self.rootdir = rootdir
        self.chunklen = chunklen
    
    def get_symbol_trades_and_quotes(self, symbol, start_ts=None, end_ts=None,
                                     as_pandas_dataframe=False):
        
        start_ts = pd.to_datetime(start_ts)
        end_ts   = pd.to_datetime(end_ts)
        #read trades
        path = os.path.join(self.rootdir,symbol+'.trades')
        if not os.path.exists(path):
            raise ValueError('{} does not exist in database'.format(path))
        tsframe = TsFrame.read(path)
        if start_ts is None:
            start_ts = tsframe.min_timestamp
        if end_ts is None:
            end_ts   = tsframe.max_timestamp+pd.tseries.offsets.Milli(1)
        trades = tsframe.read_range(start_ts, end_ts, as_pandas_dataframe=as_pandas_dataframe)
        
        #read quotes
        path = os.path.join(self.rootdir,symbol+'.quotes')
        if not os.path.exists(path):
            raise ValueError('{} does not exist in database'.format(path))
        tsframe = TsFrame.read(path)
        if start_ts is None:
            start_ts = tsframe.min_timestamp
        if end_ts is None:
            end_ts   = tsframe.max_timestamp+pd.tseries.offsets.Milli(1)
        quotes = tsframe.read_range(start_ts, end_ts, as_pandas_dataframe=as_pandas_dataframe)
        
        return trades, quotes
    
    def __getitem__(self, symbol):
        #read trades
        path = os.path.join(self.rootdir, symbol+'.trades')
        if not os.path.exists(path):
            raise ValueError('{} does not exist in database'.format(path))
        trades = TsFrame.read(path)
        #read quotes
        path = os.path.join(self.rootdir, symbol+'.quotes')
        if not os.path.exists(path):
            raise ValueError('{} does not exist in database'.format(path))
        quotes = TsFrame.read(path)
        return trades, quotes
        
    def add_ticker_data(self, symbol, trades, quotes, copy=True):
        #add trades
        path = os.path.join(self.rootdir,symbol+'.trades')
        if os.path.exists(path):
            tsframe = TsFrame.read(path)
        else:
            tsframe = TsFrame(path, str_maxsize=self.trades_str_maxsize, 
                              chunklen=self.chunklen)
        tsframe.append(trades)
            
        #add qutoes
        path = os.path.join(self.rootdir,symbol+'.quotes')
        if os.path.exists(path):
            tsframe = TsFrame.read(path)
        else:
            tsframe = TsFrame(path, str_maxsize=self.quotes_str_maxsize, 
                              chunklen=self.chunklen)
        tsframe.append(quotes)
    
    @property
    def symbols(self):
        tickers = []
        for filename in os.listdir(self.rootdir):
            if filename.endswith(".trades"):
                tickers.append( filename.split('.')[0] )
        return tickers