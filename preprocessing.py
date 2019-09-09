import tqdm
import numpy as np
import os
import datetime
import pickle
import pyarrow.parquet as parquet
def parse_set(data,uvals=None):
    if(uvals is None):
        uvals = set()
        for i in data[:100000]:
            for j in i:
                if not (j in uvals):
                    uvals.add(j)
        uvals = np.array(list(uvals))
    vals = np.zeros((len(data),len(uvals)),dtype=np.uint8)
    for i in range(vals.shape[0]):
        d = set(data[i])
        for j in range(vals.shape[1]):
            if(uvals[j] in d):
                vals[i,j] = 1
    return vals,uvals
                
def set_proper_int_dtype(data):
    if(data.min() >=0):
        if(data.max() < 256):
            data = data.astype(np.uint8)
        elif(data.max() < 2**16):
            data = data.astype(np.uint16)
        elif(data.max() < 2**32):
            data = data.astype(np.uint32)
        else:
            data = data.astype(np.uint64)
    else:
        if(data.max() < 2**14 and data.min() > -2**14):
            data = data.astype(np.int16)
        elif(data.max() < 2**30 and data.min() > -2**30):
            data = data.astype(np.int32)
        else:
            data = data.astype(np.int64)
    return data

def one_hot_str(data):
    unames,data = np.unique(data,return_inverse=True)
    return set_proper_int_dtype(data),unames

def preprocess_regression(data,**kwargs):
    fm = np.isfinite(data)
    if(fm.mean() < 1e-5 and len(kwargs) == 0):
        kwargs['all_nan'] = True
        data[:] = -1
    elif(len(kwargs) == 0 or kwargs['all_nan']==False):
        kwargs['all_nan'] = False
        
        if not('clip' in kwargs):
            y,x = np.histogram(np.abs(data[fm]),bins=100)
            inx = len(y)-1
            while y[inx] < 1e-3:
                inx-=1            
            kwargs['clip'] = x[inx+1] if len(y) > inx+1 else x.max()
        data[fm] = np.clip(data[fm],-kwargs['clip'], kwargs['clip'])

        if not('use_log' in kwargs):
            x = np.sort(np.random.choice(np.abs(data[fm]),100000))
            kwargs['use_log']= ((x[-10] > x[10]*1e4)and(x.max() > 1e2))
        if(kwargs['use_log']):
            data[fm] = np.sign(data[fm])*np.log10(np.abs(data[fm])+1)
        
    if not('nan' in kwargs):
        kwargs['nan'] = -1
    data[fm==False] = kwargs['nan']
    kwargs['type'] = 'regression'
    return data.astype(np.float16),kwargs

def preprocess_categorical(data,**kwargs): 
    kwargs['type'] = 'categorical'
    nan_val = None
    if kwargs.get('all_nan',False):
        return np.zeros_like(data,dtype=np.uin8)+kwargs['nan'],kwargs
    
    if not np.issubdtype(data.dtype, np.number):
        data = data.astype(np.str)
    if np.issubdtype(data.dtype, np.number) and np.isfinite(data).sum() < len(data):
        if np.isfinite(data).sum()==0:
            kwargs['all_nan']=True
            kwargs['nan'] = kwargs.get('nan',0)
            return np.zeros_like(data,dtype=np.uint8)+kwargs['nan'],kwargs
        
        nan_val = kwargs['nan'] if('nan' in kwargs) else data[np.isfinite(data)].max()+1
        data[np.isfinite(data)==False] = nan_val
        uvals = np.sort(np.unique(np.concatenate((data[np.isfinite(data)],np.array([nan_val])))))
    else:
        uvals = np.sort(np.unique(data))
        
    unames = kwargs.get('unames', uvals)
    assert len(np.intersect1d(unames,uvals))==len(uvals),'new label found'+str(uvals)+ ' '+str(unames)
    return set_proper_int_dtype(np.searchsorted(unames,data)),{'unames':unames,'nan':nan_val,'type':'categorical'}

def preprocess_time(data,**kwargs):
    dt = np.zeros((len(np.abs(data)),3),dtype=np.uint16)
    for i,d in enumerate(data):
        if(np.isfinite(d)):
            d = datetime.datetime.fromtimestamp(d/1000)
            dt[i,0] =(d - datetime.datetime(d.year,1,1)).days
            dt[i,1] = d.isoweekday()
            dt[i,2] = d.hour
        else:
            dt[i] = np.array([366,32,8])
    return dt,{'type':'time'}

def preprocess_onehot(data,**kwargs):
    if('unames' in kwargs):
        data,unames = parse_set(data,kwargs['unames'])
    else:
        data,unames = parse_set(data)
    return data,{'unames':unames,'type':'onehot'}

def preprocess_id(data,**kwargs):
    if(len(kwargs) == 0):
        kwargs['type'] = 'id'
        unames,cnts = np.unique(data,return_counts=True)
        kwargs['unames'] = np.sort(unames[cnts>10])
        kwargs['unknown'] = len(kwargs['unames'])+1
    inx = np.searchsorted(kwargs['unames'],data)
    inx = np.clip(inx,0,len(kwargs['unames'])-1)
    inx[data!=kwargs['unames'][inx]] = kwargs['unknown']
    return set_proper_int_dtype(inx),kwargs

def get_data_type(data,name,**kwargs):
    if('type' in kwargs):
        return kwargs['type']
    if(type(data[0]) == np.ndarray):
        return 'onehot'    
    if (data.dtype==object or 'Type' in name):
        return 'categorical'    
    if(data.dtype==np.int64 and np.abs(data).min() > 1017950800063) or 'time' in name or 'date' in name.lower():
        return 'time'
    if name.endswith('Id') or '_ID_' in name:
        return 'id'
    if '_num' in name:
        return 'regression'
    if name.endswith('Type'):
        return 'categorical'
    
    udata = np.unique(data[np.isfinite(data)])
    if(len(udata) > 7):
        return 'regression'
    else:
        return 'categorical'
    
def preprocess(data,name,**kwargs):
    dt = get_data_type(data,name,**kwargs)
    if(dt == 'time'):
        return preprocess_time(data,**kwargs)
    elif(dt == 'regression'):
        return preprocess_regression(data,**kwargs)
    elif(dt == 'categorical'):
        return preprocess_categorical(data,**kwargs)
    elif(dt == 'id'):
        return preprocess_id(data,**kwargs)
    elif(dt=='onehot'):
        return preprocess_onehot(data,**kwargs)
    else:
        assert False,'unknown data type '+dt
        

def preprocess_parquet_table(input_path,out_path,cols,params_path=None):
    for c in tqdm.tqdm(cols):
        print('preprocess ',c,end='')
        if os.path.exists(out_path+'/'+c+'.npy'):
            print('')
            continue
        data = parquet.read_table(input_path,columns=[c]).to_pandas()
        data = data[c].values
        if(params_path is None):
            data,params=preprocess(data,c)
            print(' '+params['type'])
            if params.get('all_nan',False) or data.min()==data.max():
                print('skip',c)
                continue
            np.save(out_path+'/'+c+'.npy',data)
            pickle.dump(params,open(out_path+'/'+c+'.pkl','wb'))
        else:   
            if(os.path.exists(params_path+'/'+c+'.pkl')):
                params = pickle.load(open(params_path+'/'+c+'.pkl','rb'))
                data,params=preprocess(data,c,**params)
                np.save(out_path+'/'+c+'.npy',data)
                pickle.dump(params,open(out_path+'/'+c+'.pkl','wb'))