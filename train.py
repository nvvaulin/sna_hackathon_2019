import mxnet as mx
import logging
from callbacks import Log2file
import os


def get_lr_scheduler(epoch_size, lr,
                     lr_factor, step_epochs,
                     begin_epoch=0):
    new_lr = lr
    for s in step_epochs:
        if begin_epoch >= s:
            new_lr *= lr_factor
    if new_lr != lr:
        logging.info('Adjust learning rate to %e for epoch %d' %
                     (new_lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]

    if len(steps) > 0:
        scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps,
                                                         factor=lr_factor)
    else:
        scheduler = None
    return (new_lr, scheduler)

def save_model(model_prefix):
    print('Saving with prefix:', model_prefix)
    dst_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    return mx.callback.do_checkpoint(model_prefix)


def train(sym,train_iter,val_iter,
          ctx=[mx.gpu(0),mx.gpu(1)],
          num_epoch=100,
          exp_dir='exps',
          load_epoch=9,
          name='exp1',
          lr = 0.2,
          opt='SGD',
          batch_size=512):
    if not os.path.exists(exp_dir+'/'+name+'/snapshots'):
        os.makedirs(exp_dir+'/'+name+'/snapshots')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    checkpoint = save_model(exp_dir+'/'+name+'/snapshots/'+name)

    speedometer_callback = Log2file(
                                    filename=os.path.join(exp_dir+'/'+name+'/train.log'),
                                    batch_size=batch_size,
                                    frequent=500)

    if load_epoch > 0:
        _,arg_params,aux_params = mx.model.load_checkpoint(exp_dir+'/'+name+'/snapshots/'+name,load_epoch)
    else:
        arg_params,aux_params = None,None
        
    model = mx.mod.Module(sym,
                           data_names=[i[0] for i in train_iter.provide_data],
                           label_names=[i[0] for i in train_iter.provide_label],
                           context=ctx,
                           logger=speedometer_callback.logger)


    initializer = mx.init.Xavier(rnd_type='gaussian',
                                 factor_type="in",
                                 magnitude=2)


    metrics = [mx.metric.Accuracy()]

    start_learning_rate = lr 
    schedule_multiplier = 0.32
    schedule_epochs = [3, 10, 20, 40, 65, 90, 120]
    learning_rate, learning_rate_schedule = get_lr_scheduler(len(train_iter.inx) // batch_size,
                                                             start_learning_rate,
                                                             schedule_multiplier,
                                                             schedule_epochs,
                                                             load_epoch)

    if opt == 'SGD':
        optimizer_params = dict(
            learning_rate=learning_rate,
            momentum=0.9,
            wd=1e-4,
            lr_scheduler=learning_rate_schedule
        )
        
    try:
        model.fit(
            mx.io.PrefetchingIter(train_iter),
            begin_epoch=load_epoch,
            num_epoch=num_epoch,
            eval_data=mx.io.PrefetchingIter(val_iter),
            eval_metric=metrics,
            optimizer=opt,
            optimizer_params=optimizer_params,
            initializer=initializer,
            batch_end_callback=[speedometer_callback],
            epoch_end_callback=checkpoint,
            allow_missing=True,
            arg_params=arg_params,
            aux_params=aux_params,
            kvstore='local'
        )
    except KeyboardInterrupt as e:
        return model
    return model
