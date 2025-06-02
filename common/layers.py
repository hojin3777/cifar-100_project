# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class BatchNorm2d:
    """
    2D 입력에 대한 배치 정규화 레이어
    forward 시:
      학습 시에는 미니배치의 평균과 분산으로 정규화하고, 이동 평균(running_mean)과 이동 분산(running_var)을 업데이트
      추론 시에는 저장된 이동 평균과 이동 분산을 사용하여 정규화
    backward 시:
      gamma, beta 및 입력 x에 대한 그래디언트를 계산
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, eps=1e-5):
        self.gamma = gamma  # 스케일 파라미터 (학습 대상)
        self.beta = beta    # 시프트 파라미터 (학습 대상)
        self.momentum = momentum
        self.input_shape = None # 입력 데이터의 형상 (N, C, H, W)

        # 추론 시 사용할 이동 평균/분산
        self.running_mean = running_mean
        self.running_var = running_var
        self.eps = eps # 분모가 0이 되는 것을 방지하기 위한 작은 값

        # 역전파 시 중간 계산 값 저장
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 4:
            # 입력 데이터가 4차원이 아닐 경우 예외 처리
            raise ValueError("Input must be a 4D (N, C, H, W) for BatchNorm2d.")

        if self.running_mean is None:
            N, C, H, W = x.shape
            self.running_mean = np.zeros(C, dtype=x.dtype)
            self.running_var = np.zeros(C, dtype=x.dtype)

        if train_flg:
            # 학습 시: 미니배치 통계 사용 및 이동 평균/분산 업데이트
            # (N, C, H, W) -> (N*H*W, C)로 변경하여 채널별 평균/분산 계산 용이하게
            xc_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, self.input_shape[1])
            mu = np.mean(xc_reshaped, axis=0)
            var = np.var(xc_reshaped, axis=0)

            self.xc = (x - mu.reshape(1, -1, 1, 1)) # 채널별 평균을 브로드캐스팅하여 뺌
            self.std = np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
            xn = self.xc / self.std

            self.batch_size = x.shape[0] # 역전파 시 사용
            # self.xc = (x - mu) / self.std # 이 부분은 위에서 이미 계산됨

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 추론 시: 저장된 이동 평균/분산 사용
            mu = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            xc = x - mu
            xn = xc / np.sqrt(var + self.eps)

        out = self.gamma.reshape(1, -1, 1, 1) * xn + self.beta.reshape(1, -1, 1, 1)
        return out

    def backward(self, dout):
        # (N, C, H, W) 형태의 dout을 가정
        # dbeta: dout의 각 채널별 합 (H, W 축에 대해) 후 평균
        self.dbeta = np.sum(dout, axis=(0, 2, 3))

        # dgamma: (dout * xn)의 각 채널별 합 (H, W 축에 대해) 후 평균
        # xn은 forward 시 계산된 정규화된 입력 (self.xc / self.std)
        xn = self.xc / self.std
        self.dgamma = np.sum(xn * dout, axis=(0, 2, 3))

        # dxn: dout * gamma
        dxn = self.gamma.reshape(1, -1, 1, 1) * dout

        # dxc: dxn / std
        dxc = dxn / self.std

        # dstd: sum(dxn * xc * (-1/std^2)) = -sum(dxn * xc) / std^2
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=(0, 2, 3)).reshape(1, -1, 1, 1)

        # dvar: dstd * (1/(2*sqrt(var+eps))) = dstd / (2 * std)
        dvar = dstd / (2 * self.std)

        # dmu: -sum(dxc) - dvar * (2/N*H*W) * sum(-xc)  (복잡하므로 간소화된 공식 사용)
        # 또는 dx = (1/std)[dxn - mean(dxn) - xn*mean(dxn*xn)]
        # 여기서는 일반적인 배치정규화 역전파 공식을 따름
        # (N, C, H, W) -> (N*H*W, C)
        dx_reshaped = (1. / (self.batch_size * self.input_shape[2] * self.input_shape[3])) * \
                      (self.std.reshape(1, -1, 1, 1)**-1) * \
                      ( (self.batch_size * self.input_shape[2] * self.input_shape[3]) * dxn - \
                        np.sum(dxn, axis=(0,2,3)).reshape(1,-1,1,1) - \
                        xn * np.sum(dxn * xn, axis=(0,2,3)).reshape(1,-1,1,1) )

        return dx_reshaped
    
class GlobalAveragePooling:
    """Global Average Pooling 레이어"""
    def __init__(self):
        self.params, self.grads = [], [] # 학습 파라미터 없음
        self.cache = None

    def forward(self, x):
        # x: (N, C, H, W)
        self.cache = x.shape 
        out = np.mean(x, axis=(2, 3)) # (N, C)
        return out

    def backward(self, dout):
        # dout: (N, C)
        N, C, H, W = self.cache
        # 각 채널의 평균에 대한 그래디언트를 H*W 개의 요소에 균등하게 분배
        dx_avg = dout.reshape(N, C, 1, 1) / (H * W) 
        dx = np.tile(dx_avg, (1, 1, H, W)) # (N, C, H, W)
        return dx