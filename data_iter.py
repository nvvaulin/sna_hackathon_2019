import mxnet as mx
import os
import pickle
import numpy as np

class DataIter(mx.io.DataIter):
    def __init__(self,data_path,batch_size,data_names,label_names=[],shuffle=False,nan_aug=1e-4,max_len=None,usampling=False):
        self.data,self.params = self.load_data(data_path,data_names+label_names)
        self.nan_aug = nan_aug
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.provide_data = [(i,(self.batch_size,)+self.data[i].shape[1:]) for i in data_names]
        self.provide_label = [(i,(self.batch_size,)) for i in label_names]

        self.max_len = max_len
        self.usampling = usampling
        self.reset()
        
    def load_data(self,path,names):
        params = dict([(i,pickle.load(open(path+'/'+i+'.pkl','rb'))) for i in names])
        data = {}
        for i in names:
            v = np.load(path+'/'+i+'.npy')
            if len(v.shape)==1:
                v=v.reshape((-1,1))
            data[i] = v
        return data,params
    
    def reset(self):
        self.inx = np.arange(len(self.data[self.provide_data[0][0]]),dtype=np.uint64)
        if(self.usampling):
            label = self.data[self.provide_label[0][0]].flatten()
            ulabel,cts = np.unique(label,return_counts=True)
            self.inx = np.concatenate([np.random.choice(self.inx[label==l],cts.min(),replace=False) for l in ulabel])
        if(self.shuffle):
            np.random.shuffle(self.inx)
        if not(self.max_len is None):
            self.inx=self.inx[:self.max_len]
        self.cur = 0
    
    def preprocess(self,data,params):
        data=data.copy()
        if(params['type'] == 'regression' or params['type'] == 'onehot'):
            data = np.clip(data.astype(np.float32),-1,100)
        else:
            data = data.astype(np.int32)
        if(params['type'] == 'id'):
            mask = np.random.rand(len(data))< self.nan_aug
            data[mask] = params['unknown']
        data[np.isfinite(data)==False] = 0
        return data
    
    def get_feature_size(self, key):
        params = self.params[key]
        if(params['type'] == 'time'):
            return (367,33,9),'embedding'
        elif(params['type'] == 'regression'):
            return 1,'number'
        elif(params['type'] == 'onehot'):
            return self.data[key].shape[-1],'number'
        elif(params['type'] == 'categorical'):
            return len(self.params[key]['unames']),'embedding'
        elif(params['type'] == 'id'):
            return len(self.params[key]['unames']+1),'embedding'
        
    def gen_minibatch(self):
        inx = self.inx[self.cur:self.cur+self.batch_size]
        self.cur += self.batch_size
        if(len(inx) < 1):
            raise StopIteration
        batch = {}        
        for k,v in self.data.items():
            batch[k] = self.preprocess(v[inx],self.params[k])
        return batch
    
    def next(self):
        batch = self.gen_minibatch()
        data = []
        label = []
        for i in self.provide_data:
            v = mx.nd.zeros(i[1])
            l = len(batch[i[0]])
            v[:l] = batch[i[0]]
            data.append(v)
        for i in self.provide_label:
            v = mx.nd.zeros(i[1])
            l = len(batch[i[0]])
            v[:l] = batch[i[0]].flatten()
            label.append(v)
        return mx.io.DataBatch(data=data,label=label,pad=(self.batch_size-l))
    
    
    

class SeqDataIter(mx.io.DataIter):
    def __init__(self,data_path,batch_size,data_names,label_names=[],shuffle=False,nan_aug=1e-4,max_len=None,usampling=False,seq_len=8):
        self.data,self.params = self.load_data(data_path,data_names+label_names)
        self.nan_aug = nan_aug
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.provide_data = [(i,(self.batch_size,)+self.data[i].shape[1:]) for i in data_names]
        self.provide_label = [(i,(self.batch_size,)) for i in label_names]

        self.max_len = max_len
        self.usampling = usampling
        self.reset()
        
    def load_data(self,path,names):
        params = dict([(i,pickle.load(open(path+'/'+i+'.pkl','rb'))) for i in names])
        data = {}
        for i in names:
            v = np.load(path+'/'+i+'.npy')
            if len(v.shape)==1:
                v=v.reshape((-1,1))
            data[i] = v
        return data,params
    
    def reset(self):
        self.inx = np.arange(len(self.data[self.provide_data[0][0]]),dtype=np.uint64)
        if(self.usampling):
            label = self.data[self.provide_label[0][0]].flatten()
            ulabel,cts = np.unique(label,return_counts=True)
            self.inx = np.concatenate([np.random.choice(self.inx[label==l],cts.min(),replace=False) for l in ulabel])
        if(self.shuffle):
            np.random.shuffle(self.inx)
        if not(self.max_len is None):
            self.inx=self.inx[:self.max_len]
        self.cur = 0
    
    def preprocess(self,data,params):
        data=data.copy()
        if(params['type'] == 'regression' or params['type'] == 'onehot'):
            data = np.clip(data.astype(np.float32),-1,100)
        else:
            data = data.astype(np.int32)
        if(params['type'] == 'id'):
            mask = np.random.rand(len(data))< self.nan_aug
            data[mask] = params['unknown']
        data[np.isfinite(data)==False] = 0
        return data
    
    def get_feature_size(self, key):
        params = self.params[key]
        if(params['type'] == 'time'):
            return (367,33,9),'embedding'
        elif(params['type'] == 'regression'):
            return 1,'number'
        elif(params['type'] == 'onehot'):
            return self.data[key].shape[-1],'number'
        elif(params['type'] == 'categorical'):
            return len(self.params[key]['unames']),'embedding'
        elif(params['type'] == 'id'):
            return len(self.params[key]['unames']+1),'embedding'
        
    def gen_minibatch(self):
        inx = self.inx[self.cur:self.cur+self.batch_size]
        self.cur += self.batch_size
        if(len(inx) < 1):
            raise StopIteration
        batch = {}        
        for k,v in self.data.items():
            batch[k] = self.preprocess(v[inx],self.params[k])
        return batch
    
    def next(self):
        batch = self.gen_minibatch()
        data = []
        label = []
        for i in self.provide_data:
            v = mx.nd.zeros(i[1])
            l = len(batch[i[0]])
            v[:l] = batch[i[0]]
            data.append(v)
        for i in self.provide_label:
            v = mx.nd.zeros(i[1])
            l = len(batch[i[0]])
            v[:l] = batch[i[0]].flatten()
            label.append(v)
        return mx.io.DataBatch(data=data,label=label,pad=(self.batch_size-l))