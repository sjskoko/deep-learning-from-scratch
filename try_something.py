import imp
import sys, os
from common.layers import Convolution
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np
from common.layers import Convolution as Conv
from random import random
from matplotlib import pyplot as plt

# conv layer 이해
x1 = np.arange(294).reshape(2, 3, 7, 7)
col1 = im2col(x1, 5, 5)
col1.shape

filter = np.arange(300).reshape(4, 3, 5, 5)
bias = np.zeros(10).reshape(10, 1, 1)

col_w = filter.reshape(4, -1).T
for i in range(75):
    for j in range(4):
        if i%4 == j:
            col_w[i, j] = 1
        else:
            col_w[i, j] = 0

out = np.dot(col1, col_w)
out.shape
out = out.reshape(2, 3, 3, -1)

# pooling layer 이해
pool_h = 5
pool_w = 5
H = 7
W = 7
N = 2
C = 3
out_h = (H-pool_h) +1
out_w = (W-pool_w) +1


x1 = np.arange(N*C*H*W).reshape(N, C, H, W)
col1 = im2col(x1, pool_h, pool_w) # (N*out_h*out_w, C*poo_h*pool_w)
col1 = col1.reshape(-1, pool_h*pool_w) # (N*out_h*out_w*C, poo_h*pool_w) # channel 만 변경
col1.shape

out = np.max(col1, axis=1)

out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) 

######################################################################

# SGD 이해
class SGD:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

# 모멘텀 이해
class Momentum:
    def __init__(self, lr=0.01, momentom=0.9) -> None:
        self.lr = lr
        self.momentom = momentom
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentom*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

# AdaGrad 이해
class AdaGrad:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = grads[key]**2
            params[key] -= self.lr * grads[key] /(np.sqrt(self.h[key]) + 1e-7)

# AdaGrad 이해
class AdaGrad:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = grads[key]**2
            params[key] -= self.lr * grads[key] /(np.sqrt(self.h[key]) + 1e-7)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


def func(x, y):
    return 0.01*x**2 + y**2

def x_grad(x):
    return 0.02*x 

def y_grad(y):
    return 2*y

def draw_plot(x=-7.0, y=2.0, lr=.01, optimizer = 'adam'):
    params = {'x':x, 'y':y}
    grads = {'x':0, 'y':0}
    params_track = {'x':[x], 'y':[y]}
    grads_track = {'x':[], 'y':[]}

    if optimizer == 'sgd':
        optimizer = SGD(lr)
    if optimizer == 'momentum':
        optimizer = Momentum(lr)
    if optimizer == 'adagrad':
        optimizer = AdaGrad(lr)
    if optimizer == 'adam':
        optimizer = Adam(lr)


    for i in range(10000):
        z = func(x, y)
        grads['x'] = x_grad(params['x'])
        grads['y'] = y_grad(params['y'])
        
        optimizer.update(params, grads)

        params_track['x'].append(params['x'])
        params_track['y'].append(params['y'])
        grads_track['x'].append(grads['x'])
        grads_track['y'].append(grads['y'])

    plt.plot(np.array(params_track['x']), np.array(params_track['y']))
    plt.show()

draw_plot(lr=0.01, optimizer='sgd')
draw_plot(lr=0.01, optimizer='momentum')
draw_plot(lr=0.01, optimizer='adagrad')
draw_plot(lr=0.01, optimizer='adam')



