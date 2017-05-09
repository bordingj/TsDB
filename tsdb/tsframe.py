


import pandas as pd
import bcolz
import numpy as np
import sys
import numba as nb

@nb.jit( nopython=True, nogil=True)
def assert_increasing(a):
    out = True
    for i in range(a.size-1):
        if (a[i+1]-a[i]) <= 0:
            out = False
            break
    return out

@nb.jit( nopython=True, nogil=True)
def search_sorted(a, v, from_left=True):
    if from_left:
        out_idx = 0
        for i in range(a.size):
            if a[i] >= v:
                out_idx = i
                break
    else:
        out_idx = a.size-1
        for i in range(a.size-1,-1,-1):
            if a[i] < v:
                out_idx = i+1
                break
    return out_idx

@nb.jit( nopython=True, nogil=True)
def get_max_chunk_ts(timestamps, chunk_len):
    
    max_chunk_ts = np.empty((timestamps.shape[0]//chunk_len,), dtype=np.int64)
    k = 0
    T = timestamps.shape[0]-1
    for i in range(max_chunk_ts.shape[0]):
        k += chunk_len
        max_chunk_ts[i] = timestamps[min(k-1, T)]
    return max_chunk_ts

@nb.jit( nopython=True, nogil=True)
def get_min_max_chunk_idx(max_chunk_ts, start, end, nchunks):
    min_chunk_idx = nchunks
    for chunk_idx in range(nchunks):
        if start <= max_chunk_ts[chunk_idx]:
            min_chunk_idx = chunk_idx
            break
        
    if end > max_chunk_ts[nchunks-1]:
        max_chunk_idx = nchunks
    else:
        for chunk_idx in range(min_chunk_idx,nchunks):
            if end <= max_chunk_ts[chunk_idx]:
                max_chunk_idx = chunk_idx
                break
    return min_chunk_idx, max_chunk_idx
            
class TsFrame(object):
    
    """
    On-disk Timeseries dataframe.
    """
    def __init__(self, rootdir, str_maxsize=3, chunklen=131072):
        self.rootdir = rootdir
        self.str_maxsize = str_maxsize
        self.chunklen = chunklen
        
    def _fromdataframe(self, df, **kwargs):

        # Use the names in kwargs, or if not there, the names in dataframe
        if 'names' in kwargs:
            names = kwargs.pop('names')
        else:
            names = list(df.columns.values)

        # Build the list of columns as in-memory numpy arrays and carrays
        # (when doing the conversion object -> string)
        cols = []
        # Remove a possible rootdir argument to prevent copies going to disk
        ckwargs = kwargs.copy()
        if 'rootdir' in ckwargs:
            del ckwargs['rootdir']
        for key in names:
            vals = df[key].values  # just a view as a numpy array
            if vals.dtype == np.object:
                inferred_type = pd.lib.infer_dtype(vals)
                if inferred_type == 'unicode':
                    dtype = 'U%d' % self.str_maxsize, 
                elif inferred_type == 'string':
                    # In Python 3 strings should be represented as Unicode
                    dtype = "U" if sys.version_info >= (3, 0) else "S"
                    dtype = '%s%d' % (dtype, self.str_maxsize)
                else:
                    dtype = inferred_type
            else:
                dtype = vals.dtype
            
            col = bcolz.carray(vals, dtype=dtype,
                               chunklen=self.chunklen,
                               **ckwargs)
            cols.append(col)

        # Create the ctable
        ct = bcolz.ctable(cols, names, **kwargs)
        return ct
        
    def append(self, df):
        
        #assert df is a dataframe
        assert isinstance(df, pd.DataFrame)
        
        #assert dtype of index is np.datetime64[ns]
        if df.index.dtype != '<M8[ns]':
            raise TypeError('index of dataframe must be datetime64[ns]')
        
        #convert to integer
        timestamps_as_int = df.index.astype(np.int64)

        # We also need to confirm that the rows are increasing by timestamp. This is an additional
        # constraint of TsFrame.
        if not assert_increasing(timestamps_as_int):
            raise ValueError("timestamp column must be sorted in ascending order.")
        
        
        if hasattr(self, 'ct'):
            # assert that new rows to be added are with timestamps AFTER the latest current stamp
            if self.ct['timestamp'][-1] >= timestamps_as_int[0]:
                raise ValueError('New data being appedend must have be NEWER (have timestamps after) than that present')
            if not all([x == y for x, y in zip(self.column_names, df.columns)]):
                raise ValueError('Column names of new data being appended do not match the existing column names')
            if not all([x == y for x, y in zip(self.column_dtypes, df.dtypes)]):
                raise TypeError('Column dtypes of new data being appended do not match the existing column dtypes')
            
            cols = []
            for name in self.ct.names:
                if name == 'timestamp':
                    cols.append(timestamps_as_int)
                else:
                    cols.append(df.loc[:,name].values)
            # save_prev_leftover_array
            n_chunks_prev = self.ct['timestamp'].nchunks
            timestamps_as_int = np.concatenate( (self.ct['timestamp'].leftover_array[:self.ct['timestamp'].nleftover],
                                                 timestamps_as_int)
                                                 )
            # append dataframe to ctable
            self.ct.append(cols)
            
            timestamps_as_int = timestamps_as_int[:(self.chunklen*self.ct['timestamp'].nchunks-self.chunklen*n_chunks_prev)]
            
            if len(timestamps_as_int) >= self.chunklen:
                max_chunk_ts = get_max_chunk_ts(timestamps_as_int, 
                                                self.chunklen)
                self.ct.attrs['max_chunk_ts'] += max_chunk_ts.tolist()
        else:
            
            ct_new = self._fromdataframe(df, rootdir=self.rootdir)
            ct_new.addcol(timestamps_as_int, name='timestamp', chunklen=self.chunklen)
            self.ct = ct_new
            self.column_names = df.columns.tolist()
            self.column_dtypes = df.dtypes.tolist()
            if len(timestamps_as_int) >= self.chunklen:
                max_chunk_ts = get_max_chunk_ts(timestamps_as_int, 
                                                                    self.chunklen)
                self.ct.attrs['max_chunk_ts'] = max_chunk_ts.tolist()
            else:
                self.ct.attrs['max_chunk_ts'] = []
            self.ct.attrs['str_maxsize'] = self.str_maxsize
            self.ct.attrs['chunklen']    = self.chunklen
            self.ct.attrs['column_names'] = self.column_names                
        
        assert len(self.ct.attrs['max_chunk_ts']) == self.ct['timestamp'].nchunks
        if self.ct['timestamp'].nchunks > 1:
            assert self.ct.attrs['max_chunk_ts'][-1]  == self.ct['timestamp'].chunks[self.ct['timestamp'].nchunks-1][self.chunklen-1]
            assert self.ct.attrs['max_chunk_ts'][0]  == self.ct['timestamp'].chunks[0][self.chunklen-1]
        elif self.ct['timestamp'].nchunks == 1:
            assert self.ct.attrs['max_chunk_ts'][0]  == self.ct['timestamp'].chunks[0][self.chunklen-1]
            
        
        self.ct.flush()
        
        self.ct.attrs['min_timestamp'] = self.ct['timestamp'][0]
        self.ct.attrs['max_timestamp'] = self.ct['timestamp'][self.ct.shape[0]-1]
        self.min_timestamp = pd.to_datetime(self.ct.attrs['min_timestamp'])
        self.max_timestamp = pd.to_datetime(self.ct.attrs['max_timestamp'])
        return self
        
    def read_range(self, start, end, as_pandas_dataframe=True):
        
        assert end > start
        if start <= self.min_timestamp and end > self.max_timestamp:
            out_df = pd.DataFrame(self.ct[self.column_names[0]][:],
                                columns=[self.column_names[0]],
                                )
            for name in self.column_names[1:]:
                out_df[name] = self.ct[name][:]
            out_df.index = self.ct['timestamp'][:].astype(np.dtype('<M8[ns]'))
            return out_df

        if self.ct.shape[0] == 0 or self.min_timestamp >= end or self.max_timestamp < start:
            return pd.DataFrame(np.empty((0,len(self.column_names)),
                                          ),
                                columns=self.column_names)
        #convert to integers                
        start = pd.to_datetime(start).asm8.astype(np.int64)
        end   = pd.to_datetime(end).asm8.astype(np.int64)
        
        timestamps = self.ct['timestamp']
        

        
        max_chunk_ts = self.ct.attrs['max_chunk_ts']
                                
        if timestamps.nleftover > 0:
            assert start < timestamps.leftover_array[timestamps.nleftover-1]
        else:
            assert start < max_chunk_ts[-1]
            
        if len(max_chunk_ts) < 1:
            start_idx = search_sorted(timestamps.leftover_array[:timestamps.nleftover], start, from_left=True)
            end_idx   = search_sorted(timestamps.leftover_array[:timestamps.nleftover], end, from_left=False)
        else:
            
            min_chunk_idx, max_chunk_idx = get_min_max_chunk_idx( max_chunk_ts, 
                                                                 start, end,
                                                               timestamps.nchunks)
    
            if min_chunk_idx != timestamps.nchunks:
                min_ts_chunk_ndarray = timestamps.chunks[min_chunk_idx][:]
                start_idx = search_sorted(min_ts_chunk_ndarray, start, from_left=True)
            else:
                start_idx = search_sorted(timestamps.leftover_array[:timestamps.nleftover], start, from_left=True)
            
            start_idx += min_chunk_idx*self.chunklen
            
            if max_chunk_idx != timestamps.nchunks:
                max_ts_chunk_ndarray = timestamps.chunks[max_chunk_idx][:]
                end_idx = search_sorted(max_ts_chunk_ndarray, end, from_left=False)
            else:
                end_idx = search_sorted(timestamps.leftover_array[:timestamps.nleftover], end, from_left=False)
            end_idx += max_chunk_idx*self.chunklen
            
        out_df = pd.DataFrame(self.ct[self.column_names[0]][start_idx:end_idx],
                            columns=[self.column_names[0]],
                            )
        for name in self.column_names[1:]:
            out_df[name] = self.ct[name][start_idx:end_idx]
            
        out_df.index = timestamps[start_idx:end_idx].astype(np.dtype('<M8[ns]'))
        return out_df
    
    @staticmethod
    def read(rootdir):
        ct   = bcolz.ctable(rootdir=rootdir)
        tsdf = TsFrame(rootdir, 
                       str_maxsize=ct.attrs['str_maxsize'], 
                       chunklen=ct.attrs['chunklen'] 
                       )
        tsdf.column_names = ct.attrs['column_names']
        tsdf.column_dtypes = [ct.dtype.fields[name][0] for name in tsdf.column_names]
        tsdf.min_timestamp = pd.to_datetime(ct.attrs['min_timestamp'])
        tsdf.max_timestamp = pd.to_datetime(ct.attrs['max_timestamp'])
        tsdf.ct = ct
        return tsdf