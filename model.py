import mxnet as mx

def make_input(name,s,t):
    def emb_size(n):
        if n < 100:
            return 16
        else:
            return 64
    if type(s) == tuple:
        data = mx.sym.Variable(name)
        data = mx.sym.split(data,len(s),axis=1)
        data = [mx.sym.Embedding(d,input_dim=i,name=name+'_emb'+str(j),output_dim=emb_size(i)).reshape((0,-1)) for j,(i,d) in enumerate(zip(s,data))]
        return mx.sym.concat(*data,dim=1)
    else:
        if(t == 'number'):
            return mx.sym.Variable(name)
        else:
            return mx.sym.Embedding(mx.sym.Variable(name),name=name+'_emb',input_dim=s,output_dim=emb_size(s)).reshape((0,-1))
        
def make_network(data_iter):
    def make_layer(data,num_features,name,dop=0.5):
        data = mx.sym.FullyConnected(data,num_hidden=num_features,name='fc'+name)
        data = mx.sym.BatchNorm(data,name='bn'+name)
        data = mx.sym.Activation(data,name='softrelu'+name,act_type='softrelu')
        return mx.sym.Dropout(data,p=dop)
    
    inputs = [make_input(i[0],*data_iter.get_feature_size(i[0])) for i in data_iter.provide_data]
    res = mx.sym.concat(*inputs,dim=1,name='total_feature')
    res = mx.sym.BatchNorm(res,name='bn1')
    res = mx.sym.Activation(res,name='softrelu1',act_type='softrelu')
    res = make_layer(res,1024,'2')
    res = make_layer(res,1024,'3')
    res = make_layer(res,512,'4')
    return mx.sym.FullyConnected(res,num_hidden=2,name='output')

def add_loss(net):
    return  mx.sym.SoftmaxOutput(net,label = mx.sym.var('liked'))
#     res = mx.sym.clip(mx.sym.sigmoid(net),1e-3,1-1e-3)
#     res = mx.sym.Group([-mx.sym.mean(mx.sym.log(res)*label+mx.sym.log(1-res)*(1-label)),mx.sym.BlockGrad(net))
#     return mx.sym.MakeLoss(res,name='loss')