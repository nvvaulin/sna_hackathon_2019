from preprocessing import preprocess_parquet_table
import pyarrow.parquet as parquet
from data_iter import DataIter
import mxnet as mx
import numpy as np
import pickle
from data_iter import SeqDataIter
import mxnet as mx
import os
from model import make_network,add_loss
from train import train
from test import predict,score,make_submission
from sacred import Experiment

def make_like_feature(path):
    data = np.load(path+'/feedback.npy')
    params = pickle.load(open(path+'/feedback.pkl','rb'))
    inx = list(params['unames']).index('Liked')
    data = data[:,[inx]]
    pickle.dump({'type':'categorical','unames':['NonLiked','Liked']},open(path+'/liked.pkl','wb'))
    np.save(path+'/liked.npy',data.astype(np.uint8))

ex = Experiment('default')

@ex.config
def config():
    preprocess_path='./data_preprocessed'
    train_path='./train_preprocessed'
    val_path='./val_preprocessed'
    test_path='./test_preprocessed'
    ctx=[mx.gpu(0),mx.gpu(1)]
    num_epoch=100
    exp_dir='exps'
    load_epoch=0
    name='seq2seq'
    batch_size=512
    seq_len = 8
    remove_features = ['instanceId_objectId','instanceId_userId','liked']
    original_data_root = './data'

@ex.automain
def preprocess_data(remove_features,label_names,train_path,val_path,preprocess_path,original_data_root):
    preprocess_parquet_table(os.path.join(original_data_root,'collabTrain'),preprocess_path,\
                             cols=parquet.read_table(os.path.join(original_data_root,'collabTrain')).to_pandas().columns)

    preprocess_parquet_table(os.path.join(original_data_root,'collabTest'),test_path,\
                             cols=parquet.read_table(os.path.join(original_data_root,'collabTest')).to_pandas().columns,\
                             params_path=preprocess_path)
    data_names=[i[:-4] for i in os.listdir(preprocess_path) if i.endswith('.npy') and not  (i[:-4] in remove_features)]
    data_iter = DataIter(preprocess_path,100,data_names=data_names,label_names=label_names,shuffle=True)
    train = data_iter.inx[1000000:]
    val = data_iter.inx[:1000000]
    for k,v in data_iter.data.items():
        np.save(os.path.join(val_path,'/%s.npy'%k),v[val])
        np.save(os.path.join(train_path,'/%s.npy'%k),v[train])
    
@ex.automain
def train_model(remove_features,label_names,train_path,val_path,batch_size,seq_len,num_epoch,ctx,load_epoch,name):
    import os
    from test import predict,score,make_submission
    data_names=[i[:-4] for i in os.listdir(train_path) if i.endswith('.npy') and not  (i[:-4] in remove_features)]
    train_iter = SeqDataIter(train_path,batch_size,
                             data_names=data_names,
                             label_names=label_names,
                             shuffle=True,
                             usampling=True,
                             seq_len=seq_len)
    val_iter = SeqDataIter(val_path,batch_size,
                           data_names=data_names,
                           label_names=label_names,
                           shuffle=False,
                           max_len=batch_size*1000,
                           seq_len=seq_len)
    
    sym = make_network(train_iter,seq_len)
    sym = add_loss(sym)

    model = train(sym,train_iter,val_iter,
          name=name,
          load_epoch=load_epoch,
          batch_size=batch_size,
          exp_dir=exp_dir)
    
    test_iter = DataIter(test_path,batch_size,data_names=data_names,label_names=[],shuffle=False)
    score(val_iter,7,name)
    prediction = predict(test_iter,7,name)    
    make_submission(prediction,name)



    
    
    