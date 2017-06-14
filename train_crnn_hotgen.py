#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys, random
sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx
import logging

from symbol.crnn import crnn

from io import BytesIO
import cv2, random
import cPickle
import os

sys.path.append('./generate_data')
from generate_data import GenTextLine

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

        self.pad = 0
        self.index = None # TODO: what is index?

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class OCRIter(mx.io.DataIter):
    def __init__(self, total_size, batch_size, classes, data_shape, num_label, init_states):
        super(OCRIter, self).__init__()
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]
        self.classes = classes
        self.count = total_size / self.batch_size
        self.provide_data = [('data', (batch_size, 1, data_shape[1], data_shape[0]))] + init_states
        self.provide_label = [('label', (self.batch_size, num_label))]
        self.G = GenTextLine(['./generate_data/fonts/Songti.ttc',
                             './generate_data/fonts/PingFang.ttc',
                             './generate_data/fonts/Kai.ttf',
                             './generate_data/fonts/STHeitiMedium.ttc'],
                             font_size=20)

    def __iter__(self):
        global words
        #print('iter')
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                plate_str = unicode(str(random.randint(0,999)))
                if random.random() < 0.25:
                    plate_str =  u'等级' + plate_str
                if random.random() < 0.5:
                    plate_str = plate_str + u'.' + unicode(str(random.randint(0,99)))
                if random.random() < 0.5 or len(plate_str) < 2:
                    plate_str = plate_str + u'%'

                if random.random() < 0.25:
                    plate_str = random.choice(words)

                plate_str = plate_str[:self.num_label] # ensure max length
                img = self.G.generate(plate_str)
                assert len(img.shape) == 2  #gray-scale

                img = cv2.resize(img, self.data_shape)
                img = img.reshape((1, data_shape[1], data_shape[0]))
                #print(img)
                #img = img.transpose(1, 0)
                #img = img.reshape((data_shape[0] * data_shape[1]))
                img = np.multiply(img, 1/255.0)
                #print(img)
                data.append(img)
                ret = np.zeros(self.num_label, int)
                for number in range(len(plate_str)):
                    ret[number] = self.classes.index(plate_str[number]) + 1
                #print(ret)
                label.append(ret)

            data_all = [mx.nd.array(data)] + self.init_state_arrays
            label_all = [mx.nd.array(label)]
            data_names = ['data'] + init_state_names
            label_names = ['label']


            data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


BATCH_SIZE = 32
SEQ_LENGTH = 25

def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        for k in range(SEQ_LENGTH):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

if __name__ == '__main__':
    # set up logger
    log_file_name = "crnn_plate.log"
    log_file = open(log_file_name, 'w')
    log_file.close()
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_name)
    logger.addHandler(fh)

    prefix = os.path.join(os.getcwd(), 'model', 'crnn_ctc')

    num_hidden = 256
    num_lstm_layer = 2

    num_epoch = 100
    learning_rate = 0.001
    momentum = 0.9
    num_label = 9
    data_shape = (100, 32)
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "%"]
    classes = [unicode(x) for x in classes]
    #words = [u"等级"]
    words = [u"等级",u"体质",u"根骨",u"力道",u"身法",u"元气",u"攻击力",u"命中",u"会心",u"会心效果",u"加速",u"破防",u"无双",u"外功防御",u"内功防御",u"闪躲",u"招架",u"拆招",u"御劲",u"化劲",u"气血回转",u"内力回转",u"跑速",u"治疗量"]
    char_set = set()
    for w in words:
        for c in w:
            char_set.add(c)
    classes = classes + list(char_set)
    num_classes = len(classes) + 1

    contexts = [mx.context.gpu(0)]

    def sym_gen(seq_len):
        return crnn(num_lstm_layer, seq_len,
                           num_hidden=num_hidden, num_classes = num_classes,
                           num_label = num_label, dropout=0.3)

    init_c = [('l%d_init_c'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer*2)]
    init_h = [('l%d_init_h'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer*2)]
    init_states = init_c + init_h

    data_train = OCRIter(80000, BATCH_SIZE, classes, data_shape, num_label, init_states)
    data_val = OCRIter(8000, BATCH_SIZE, classes, data_shape, num_label, init_states)

    symbol = sym_gen(SEQ_LENGTH)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 wd=0.00001,
                                 #optimizer='AdaDelta',
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    logger.info('begin fit')

    model.fit(X=data_train, eval_data=data_val,
              eval_metric = mx.metric.np(Accuracy),
              batch_end_callback=mx.callback.Speedometer(BATCH_SIZE, 100), logger = logger,
              epoch_end_callback = mx.callback.do_checkpoint(prefix, 1))

    model.save("crnnctc")

