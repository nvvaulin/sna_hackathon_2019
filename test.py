import numpy as np
import mxnet as mx
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class DataIterHideLabel(mx.io.DataIter):
    def __init__(self,data_iter):
        self.data_iter=data_iter
        self.provide_label=[]
    @property
    def provide_data(self):
        return self.data_iter.provide_data
    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = next(self.data_iter)
        batch.label = []
        return batch

def predict(data_iter,load_epoch,name,
          ctx=[mx.gpu(0),mx.gpu(1)],
          exp_dir='exps',
          batch_size=512):
    data_iter = DataIterHideLabel(data_iter)
    sym, arg_params, aux_params =mx.model.load_checkpoint(exp_dir+'/'+name+'/snapshots/'+name,load_epoch)
    for i in sym.get_internals():
        if(i.name=='output'):
            sym = i
        
    model = mx.mod.Module(sym,
                          data_names=[i[0] for i in data_iter.provide_data],
                          label_names=[],
                          context=ctx)
    model.bind(data_shapes=data_iter.provide_data)
    model.set_params(arg_params,aux_params)
    prediction = model.predict(data_iter)
    return prediction.asnumpy()[:,1]

def score(data_iter,load_epoch,name,
          ctx=[mx.gpu(0),mx.gpu(1)],
          exp_dir='exps',
          batch_size=512):
    label_name = val_iter.provide_label[0][0]
    true = val_iter.data[label_name][val_iter.inx]
    prediction = predict(data_iter,load_epoch,name,ctx,exp_dir,batch_size)
    tpr,fpr,th = roc_curve(true[:,0],prediction)
    plt.plot(tpr,fpr)
    plt.show()
    
def make_submission(prediction,name):
    from pyarrow import parquet
    result = parquet.read_table('data/collabTest',columns=["instanceId_userId", "instanceId_objectId"]).to_pandas()
    result['predictions'] = -prediction
    result = result.sort_values(by=['instanceId_userId', 'predictions'])
    submit = result.groupby("instanceId_userId")['instanceId_objectId'].apply(list)
    submit.to_csv("results/%s.csv.gz"%name, header = False, compression='gzip')