import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np

class Convolution:
    def __init__(self, W, b, stride=1, pad=0) -> None:
        self.W = W # filter
        self.b = b # bias
        self.stride = stride # stride
        self.pad = pad #padding

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

x= np.arange(147).reshape(1, 3, 7, 7)

col = im2col(x, 2, 2)
c0l = col.reshape(-1, 4)



class Pooling:
