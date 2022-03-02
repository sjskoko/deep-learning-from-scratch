import imp
import sys, os
from common.layers import Convolution
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np
from common.layers import Convolution as Conv

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