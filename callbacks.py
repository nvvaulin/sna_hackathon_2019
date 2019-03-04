import logging
import mxnet as mx
from mxnet.callback import Speedometer
import os
import logging
import math
import time

class Log2file(Speedometer):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        print (filename)
        self.logger = logging.getLogger('spam_application')
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(self.filename)
        self.fh.setLevel(logging.DEBUG)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)
        super(Log2file, self).__init__(**kwargs)

    def __call__(self, param):
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed = float('inf')
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                        msg = 'Epoch[%d] Batch [%d-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f' * len(name_value)
                        self.logger.info(msg, param.epoch, count - self.frequent, count, speed,  *sum(name_value, ()))
                    else:
                        msg = 'Epoch[%d] Batch [0-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f' * len(name_value)
                        self.logger.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    self.logger.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()