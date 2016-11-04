import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import nn

import layers

DATA_DIR = '/home/kivan/datasets/MNIST/'
SAVE_DIR = "/home/kivan/source/fer/out/"

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
#config['weight_decay'] = 1e-3
config['weight_decay'] = 1e-2
config['lr_policy'] = {1:{'lr':1e-2}, 3:{'lr':1e-3}, 5:{'lr':1e-4}, 7:{'lr':1e-5}}
#config['lr_policy'] = {1:{'lr':1e-3}, 3:{'lr':1e-3}, 5:{'lr':1e-4}, 7:{'lr':1e-5}}
#config['lr_policy'] = {1:{'lr':1e-4}, 3:{'lr':1e-4}, 5:{'lr':1e-4}, 7:{'lr':1e-5}}

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 1, 28, 28])
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 1, 28, 28])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 1, 28, 28])
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

weight_decay = config['weight_decay']
net = []
regularizers = []
inputs = np.random.randn(config['batch_size'], 1, 28, 28)
net += [layers.Convolution(inputs, 32, 3, "conv1")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv1_l2reg')]
net += [layers.MaxPooling(net[-1], "pool1")]
net += [layers.ReLU(net[-1], "relu1")]
net += [layers.Convolution(net[-1], 32, 5, "conv3")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'conv2_l2reg')]
net += [layers.MaxPooling(net[-1], "pool2")]
net += [layers.ReLU(net[-1], "relu3")]
## 7x7
net += [layers.Flatten(net[-1], "flatten4")]
net += [layers.FC(net[-1], 512, "fc5")]
regularizers += [layers.L2Regularizer(net[-1].weights, weight_decay, 'fc5_l2reg')]
net += [layers.ReLU(net[-1], "relu5")]
net += [layers.FC(net[-1], 10, "logits")]

data_loss = layers.SoftmaxCrossEntropyWithLogits()
loss = layers.RegularizedLoss(data_loss, regularizers)

nn.train(train_x, train_y, valid_x, valid_y, net, loss, config)
nn.evaluate("Test", test_x, test_y, net, loss, config)

