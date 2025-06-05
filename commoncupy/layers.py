try:
    import cupy as cp
    cp.cuda.Device(0).use()
    print("CuPy is available and using GPU 0.")
    xp = cp # 연산에 사용할 주 라이브러리 (CuPy)
except Exception as e:
    import numpy as np
    print(f"CuPy not available or GPU not found: <<{e}>>. Falling back to NumPy for core operations.")
    xp = np # CuPy 사용 불가 시 NumPy로 대체 (오류 방지 및 CPU 실행용)
    # 이 경우, 아래 정의되는 함수/클래스에서 xp를 사용하므로 NumPy로 동작하게 됩니다.

from commoncupy.functions import *
from commoncupy.util import *

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
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x_reshaped = x.reshape(x.shape[0], -1)
        self.x = x_reshaped # self.x를 CuPy 배열로 유지
        out = xp.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

class SoftmaxWithLoss:
    def __init__(self, smoothing_alpha=0.0): # smoothing_alpha 파라미터 추가
        self.loss = None
        self.y = None 
        self.t_original = None # 원본 타겟 (원-핫)
        self.t_for_grad = None # 그래디언트 계산에 사용될 타겟 (부드럽게 처리될 수 있음)
        self.smoothing_alpha = smoothing_alpha
        self.num_classes = None

    def forward(self, x, t, train_flg=True): # train_flg 파라미터 추가
        self.t_original = t # t는 (N,C) 형태의 원-핫 인코딩된 레이블로 가정
        self.y = softmax(x)

        if self.num_classes is None:
            self.num_classes = self.y.shape[1]

        if self.smoothing_alpha > 0 and train_flg:
            # Label smoothing 적용: t_smooth = t_one_hot * (1 - alpha) + alpha / num_classes
            # self.t_original은 이미 (N,C) 형태의 원-핫 레이블
            self.t_for_grad = self.t_original * (1.0 - self.smoothing_alpha) + (self.smoothing_alpha / self.num_classes)
        else:
            self.t_for_grad = self.t_original # 학습 중이 아니거나 alpha가 0이면 원본 사용

        self.loss = cross_entropy_error(self.y, self.t_for_grad)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t_original.shape[0] # 원본 t의 배치 크기 사용
        # 그래디언트는 y - t_target (여기서 t_target은 손실 계산에 사용된 self.t_for_grad)
        dx = (self.y - self.t_for_grad) / batch_size
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # self.mask = xp.random.rand(*x.shape) > self.dropout_ratio # CuPy의 rand는 0~1 사이의 균일 분포
            # 또는 CuPy의 random_sample 사용
            # self.mask = xp.random.random(x.shape, dtype=x.dtype) > self.dropout_ratio # 기존 코드
            if hasattr(xp.random, 'random') and 'dtype' in xp.random.random.__code__.co_varnames: # CuPy 등 dtype 지원하는 경우
                self.mask = xp.random.random(x.shape, dtype=x.dtype) > self.dropout_ratio
            else: # NumPy의 경우 dtype 인자 없음
                self.mask = xp.random.random(x.shape).astype(x.dtype, copy=False) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class BatchNormalization: # FC 레이어용 (2D 입력)
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        self.running_mean = running_mean
        self.running_var = running_var  
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
        self.eps = 1e-7 # 작은 값 추가

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2: # FC용이므로 2D로 가정
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self._forward(x, train_flg) # _forward로 변경
        
        return out.reshape(*self.input_shape) # 원래 형태로 복원

    def _forward(self, x, train_flg): # 실제 연산은 _forward에서
        if self.running_mean is None:
            # D = x.shape[1] # CuPy 배열의 shape 사용
            # self.running_mean = xp.zeros(D, dtype=x.dtype)
            # self.running_var = xp.zeros(D, dtype=x.dtype)
            # 초기화는 모델 생성 시 파라미터로 받으므로 여기서는 D를 사용한 초기화는 불필요할 수 있음
            # 만약 running_mean/var가 None으로 들어오면, 입력 x의 채널 수에 맞게 초기화
             D = x.shape[1]
             if self.running_mean is None:
                 self.running_mean = xp.zeros(D, dtype=x.dtype)
             if self.running_var is None:
                 self.running_var = xp.zeros(D, dtype=x.dtype) # 0 대신 1로 초기화하는 경우도 있음, 여기서는 0으로.
                        
        if train_flg:
            mu = xp.mean(x, axis=0)
            xc = x - mu
            var = xp.mean(xc**2, axis=0)
            std = xp.sqrt(var + self.eps) # eps 추가
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn # backward에서 사용하기 위해 xn 저장
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / (xp.sqrt(self.running_var + self.eps)) # eps 추가
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2: # FC용이므로 2D로 가정
            # N, C, H, W = dout.shape # 이 부분은 2D가 아닐 때의 처리인데, FC용 BN에서는 보통 2D로 들어옴
            dout_orig_shape = dout.shape
            dout = dout.reshape(dout_orig_shape[0], -1)

        dx = self._backward(dout) # _backward로 변경

        return dx.reshape(*self.input_shape) # 원래 형태로 복원

    def _backward(self, dout): # 실제 연산은 _backward에서
        # dbeta = xp.sum(dout, axis=0)
        # dgamma = xp.sum(self.xn * dout, axis=0) # forward에서 저장한 xn 사용
        # dxn = self.gamma * dout
        # dxc = dxn / self.std
        # dstd = -xp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        # dvar = 0.5 * dstd / self.std # 오타 수정: dstd * (1/(2*std))
        # dxc += (2.0 / self.batch_size) * self.xc * dvar # 이 부분은 평균에 대한 미분
        # dmu = xp.sum(dxc, axis=0)
        # dx = dxc - dmu / self.batch_size # 이 부분도 평균에 대한 미분
        
        # 교재의 간결한 공식 사용
        self.dbeta = xp.sum(dout, axis=0)
        self.dgamma = xp.sum(self.xn * dout, axis=0) # forward에서 저장한 xn 사용
        
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -xp.sum(dxn * self.xc, axis=0) / (self.std * self.std) # d(1/std) = -1/std^2
        dvar = 0.5 * dstd / self.std # d(sqrt(var)) = 1/(2*sqrt(var))
        
        # d(var) = d(mean(xc^2)) = (1/N) * sum(2*xc*dxc_from_var)
        # 여기서는 dvar가 d(var)에 대한 그래디언트가 아니라, std에 대한 그래디언트로부터 유도됨
        # 따라서, 교재의 공식을 따르는 것이 더 안전
        # dx = (1. / self.batch_size) * self.gamma * (self.std**-1) * \
        #      (self.batch_size * dxn - xp.sum(dxn, axis=0) - self.xn * xp.sum(dxn * self.xn, axis=0))
        
        # 더 명확한 분해 (Deep Learning from Scratch 책의 구현 방식)
        dx = (1.0 / self.batch_size) * self.gamma * (1.0 / self.std) * \
             (self.batch_size * dxn - xp.sum(dxn, axis=0) - self.xn * xp.sum(dxn * self.xn, axis=0))

        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        self.x = None   
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = conv_output_size(H, FH, self.stride, self.pad)
        out_w = conv_output_size(W, FW, self.stride, self.pad)

        self.x = x # CuPy 배열로 저장
        self.col = im2col(x, FH, FW, self.stride, self.pad) # CuPy im2col 사용
        self.col_W = self.W.reshape(FN, -1).T # (C*FH*FW, FN)

        out = xp.dot(self.col, self.col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # (N, FN, out_h, out_w)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        N, C_out, out_h, out_w = dout.shape # dout은 (N, FN, out_h, out_w)
        
        dout = dout.transpose(0,2,3,1).reshape(-1, FN) # (N*out_h*out_w, FN)

        self.db = xp.sum(dout, axis=0)
        self.dW = xp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = xp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) # CuPy col2im 사용

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
        out_h = pool_output_size(H, self.pool_h, self.stride) # int 강제 형변환은 pool_output_size에서 처리
        out_w = pool_output_size(W, self.pool_w, self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) # CuPy im2col
        col = col.reshape(-1, self.pool_h*self.pool_w)

        self.x = x # CuPy 배열 저장
        self.arg_max = xp.argmax(col, axis=1) # CuPy argmax
        out = xp.max(col, axis=1) # CuPy max
        
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        # dout: (N, C, out_h, out_w)
        dout_transposed = dout.transpose(0, 2, 3, 1) # (N, out_h, out_w, C)
        
        pool_size = self.pool_h * self.pool_w
        # dmax = xp.zeros((dout.size, pool_size), dtype=self.x.dtype) # 원본 x의 dtype 사용
        # dmax[xp.arange(self.arg_max.size), self.arg_max.flatten()] = dout_transposed.flatten()
        
        # CuPy에서 위와 같은 방식의 인덱싱 할당은 느릴 수 있음.
        # 더 효율적인 방법은 scatter_add와 유사한 연산을 사용하는 것이나,
        # 여기서는 NumPy와 유사한 방식으로 일단 구현.
        # 또는, 각 arg_max 위치에 해당하는 dout 값을 직접 dcol에 배치하는 방식.

        N, C, H, W = self.x.shape
        out_h = pool_output_size(H, self.pool_h, self.stride)
        out_w = pool_output_size(W, self.pool_w, self.stride)

        # dcol 초기화 (N * out_h * out_w, C * pool_h * pool_w) -> (N*out_h*out_w, pool_h*pool_w) for each channel
        # 실제로는 (N*out_h*out_w, C, pool_h*pool_w) 형태로 다루고 채널별로 처리하거나,
        # (N*out_h*out_w*C, pool_h*pool_w) 형태로 만들어야 함.
        # im2col의 출력은 (N*out_h*out_w, C*FH*FW) 이므로, pooling의 col은 (N*out_h*out_w*C, pool_h*pool_w)가 아님.
        # col.reshape(-1, self.pool_h*self.pool_w)는 (N*C*out_h*out_w, pool_h*pool_w) 형태가 됨.
        
        # arg_max는 (N*C*out_h*out_w) 크기의 1D 배열. 각 요소는 0 ~ pool_size-1 사이의 값.
        # dout_flattened는 (N*C*out_h*out_w) 크기의 1D 배열.
        
        # dout을 (N, C, out_h, out_w) -> (N*C*out_h*out_w) 로 flatten
        dout_flattened = dout.transpose(0,1,2,3).reshape(-1) # (N,C,OH,OW) -> (N*C*OH*OW)
        
        dcol = xp.zeros((self.arg_max.size, pool_size), dtype=self.x.dtype)
        # dcol[xp.arange(self.arg_max.size), self.arg_max] = dout_flattened # self.arg_max는 이미 1D
        # CuPy에서는 다음과 같이 하는 것이 더 안전할 수 있음 (인덱싱 배열 타입 문제 방지)
        rows = xp.arange(self.arg_max.size, dtype=xp.int32)
        cols = self.arg_max.astype(xp.int32)
        dcol[rows, cols] = dout_flattened
        
        # dcol을 (N*out_h*out_w, C*pool_h*pool_w) 형태로 col2im에 전달해야 함.
        # 현재 dcol은 (N*C*out_h*out_w, pool_h*pool_w)
        # col2im은 (N*out_h*out_w, C*FH*FW) 형태의 입력을 기대함.
        # 따라서 dcol을 (N*out_h*out_w, C, pool_h*pool_w)로 만들고,
        # (N*out_h*out_w, C*pool_h*pool_w)로 reshape해야 함.
        
        # col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad) -> (N*out_h*out_w, C*FH*FW)
        # pooling에서는 채널별로 max pooling을 수행하므로,
        # col.reshape(-1, self.pool_h*self.pool_w)는 (N*C*out_h*out_w, pool_h*pool_w)
        # arg_max도 이 형태에 맞춰짐.
        # dcol도 (N*C*out_h*out_w, pool_h*pool_w)
        
        # col2im은 (N*out_h*out_w, C*filter_h*filter_w) 형태의 입력을 받음.
        # 따라서 dcol을 (N, C, out_h, out_w, pool_h*pool_w)로 만들고,
        # (N*out_h*out_w, C*pool_h*pool_w)로 reshape해야 함.
        # dcol.reshape(N, C, out_h, out_w, -1).transpose(0,2,3,1,4).reshape(N*out_h*out_w, C*pool_h*pool_w)
        
        # col2im의 입력 col은 (N*out_h*out_w, C*filter_h*filter_w)
        # 현재 dcol은 ( (N*C*out_h*out_w), pool_h*pool_w )
        # col2im에 맞추려면, dcol을 (N*out_h*out_w, C*pool_h*pool_w) 형태로 만들어야 함.
        # 이는 dcol을 (N, out_h, out_w, C, pool_h*pool_w)로 보고, (N*out_h*out_w, C*pool_h*pool_w)로 reshape.
        # dcol.reshape(N, out_h, out_w, C, -1) # (N, OH, OW, C, PH*PW)
        # .transpose(0,1,2,3,4) # 순서 유지
        # .reshape(N*out_h*out_w, C*self.pool_h*self.pool_w)
        
        # dcol을 (N, C, out_h, out_w, pool_h*pool_w) 형태로 먼저 복원
        dcol_reshaped = dcol.reshape(N, C, out_h, out_w, self.pool_h * self.pool_w)
        # (N, C, out_h, out_w, pool_h*pool_w) -> (N, out_h, out_w, C, pool_h*pool_w)
        dcol_transposed = dcol_reshaped.transpose(0, 2, 3, 1, 4)
        # (N*out_h*out_w, C*pool_h*pool_w)
        dcol_for_col2im = dcol_transposed.reshape(N * out_h * out_w, -1)

        dx = col2im(dcol_for_col2im, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

class BatchNorm2d: # Conv 레이어용 (4D 입력)
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None, eps=1e-5):
        self.gamma = gamma  # (C,) 형태의 CuPy 배열
        self.beta = beta    # (C,) 형태의 CuPy 배열
        self.momentum = momentum
        self.input_shape = None # (N, C, H, W)

        self.running_mean = running_mean # (C,)
        self.running_var = running_var   # (C,)
        self.eps = eps

        self.batch_size = None
        self.xc = None # (N, C, H, W)
        self.std = None # (C,) 또는 (1,C,1,1)
        self.dgamma = None
        self.dbeta = None
        self.xn = None # forward에서 저장

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape # (N, C, H, W)
        N, C, H, W = x.shape

        if self.running_mean is None:
            self.running_mean = xp.zeros(C, dtype=x.dtype)
        if self.running_var is None:
            self.running_var = xp.zeros(C, dtype=x.dtype) # 분산은 0으로 초기화하면 안됨. 1로 하거나, 첫 배치에서 계산. 여기서는 0으로.

        if train_flg:
            # (N, C, H, W) -> (N*H*W, C) 로 변경하여 채널별 평균/분산 계산 용이하게
            # 또는 (0, 2, 3) 축에 대해 평균/분산 계산
            mu = xp.mean(x, axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)
            self.xc = x - mu # (N, C, H, W)
            var = xp.mean(self.xc**2, axis=(0, 2, 3), keepdims=True) # (1, C, 1, 1)
            self.std = xp.sqrt(var + self.eps) # (1, C, 1, 1)
            self.xn = self.xc / self.std # (N, C, H, W)
            
            self.batch_size = N # 역전파 시 사용

            # 이동 평균/분산 업데이트 (채널별 스칼라 값으로 저장)
            self.running_mean = self.momentum * self.running_mean.reshape(1,C,1,1) + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var.reshape(1,C,1,1) + (1 - self.momentum) * var
            self.running_mean = self.running_mean.reshape(C) # 다시 (C,)로
            self.running_var = self.running_var.reshape(C)   # 다시 (C,)로
        else:
            # 추론 시: 저장된 이동 평균/분산 사용
            mu_run = self.running_mean.reshape(1, C, 1, 1)
            var_run = self.running_var.reshape(1, C, 1, 1)
            self.xc = x - mu_run # 추론 시 xc는 backward에 필요 없음
            self.xn = self.xc / xp.sqrt(var_run + self.eps) # 추론 시 xn도 backward에 필요 없음

        # gamma와 beta는 (C,) 형태이므로 (1,C,1,1)로 브로드캐스팅
        out = self.gamma.reshape(1, C, 1, 1) * self.xn + self.beta.reshape(1, C, 1, 1)
        return out

    def backward(self, dout):
        # dout: (N, C, H, W)
        N, C, H, W = dout.shape
        
        # dbeta, dgamma 계산 (채널별 합)
        self.dbeta = xp.sum(dout, axis=(0, 2, 3)) # (C,)
        self.dgamma = xp.sum(self.xn * dout, axis=(0, 2, 3)) # (C,)
        
        # dxn 계산
        dxn = self.gamma.reshape(1, C, 1, 1) * dout # (N, C, H, W)
        
        # dxc 계산
        dxc = dxn / self.std # self.std는 (1,C,1,1)
        
        # dstd 계산
        # dstd = -xp.sum((dxn * self.xc) / (self.std * self.std), axis=(0, 2, 3), keepdims=True) # (1,C,1,1)
        # 위 식은 d(1/std)에 대한 것이 아니라, 전체 손실에 대한 std의 미분.
        # 교재의 공식을 따름:
        dstd = -xp.sum(dxn * self.xc, axis=(0,2,3), keepdims=True) / (self.std**3) # (1,C,1,1)

        # dvar 계산
        dvar = 0.5 * dstd / self.std # (1,C,1,1)
        
        # dmu 계산 및 dx 계산 (교재 공식)
        # dx = (1/N*H*W) * (1/std) * [ (N*H*W)*dxn - sum(dxn) - (xc/std^2)*sum(dxn*xc) ]
        # 여기서 sum은 (0,2,3) 축에 대한 합
        
        # dx = (1. / (N * H * W)) * (1. / self.std) * \
        #      ( (N * H * W) * dxn - xp.sum(dxn, axis=(0, 2, 3), keepdims=True) - \
        #        (self.xc / (self.std**2)) * xp.sum(dxn * self.xc, axis=(0, 2, 3), keepdims=True) )
        
        # 더 명확한 분해 (Deep Learning from Scratch 책의 구현 방식과 유사하게 4D에 적용)
        # dx = (1. / (N*H*W)) * (self.gamma.reshape(1,C,1,1) / self.std) * \
        #      ( (N*H*W) * dxn_from_gamma_effect - \
        #        xp.sum(dxn_from_gamma_effect, axis=(0,2,3), keepdims=True) - \
        #        self.xn * xp.sum(dxn_from_gamma_effect * self.xn, axis=(0,2,3), keepdims=True) )
        # 여기서 dxn_from_gamma_effect는 dxn임.

        # dx = (1. / (N * H * W)) * (self.gamma.reshape(1,C,1,1) / self.std) * \
        #      ( (N * H * W) * dxn - \
        #        xp.sum(dxn, axis=(0,2,3), keepdims=True) - \
        #        self.xn * xp.sum(dxn * self.xn, axis=(0,2,3), keepdims=True) )
        
        # Stanford CS231n의 배치 정규화 역전파 공식을 4D에 맞게 적용
        # 1. dL/dxn = dL/dout * gamma
        #    dxn = self.gamma.reshape(1,C,1,1) * dout (이미 계산됨)
        # 2. dL/dvar = sum(dL/dxn * xc * (-1/2) * (var+eps)^(-3/2))
        #    dvar = xp.sum(dxn * self.xc * (-0.5) * (self.std**-3), axis=(0,2,3), keepdims=True) (이미 dvar로 계산됨, 부호 및 계수 확인 필요)
        #    dvar_term = xp.sum(dxn * self.xc, axis=(0,2,3), keepdims=True) * (-0.5) * (self.std**-3) # (1,C,1,1)
        dvar_term = dvar # 위에서 계산한 dvar 사용 (dvar = 0.5 * dstd / std, dstd = -sum(dxn*xc)/std^3)
                       # dvar = 0.5 * (-sum(dxn*xc)/std^3) / std = -0.5 * sum(dxn*xc) / std^4
                       # 이 부분은 CS231n 공식과 직접 매칭되지 않음.
                       # 교재의 분해된 공식을 사용하는 것이 더 안전.

        # 교재의 2D Batch Norm 역전파를 4D로 확장
        # dx = (1/M) * (gamma/std) * [M*dxn - sum(dxn) - xn*sum(dxn*xn)] where M = N*H*W
        M = N * H * W
        sum_dxn = xp.sum(dxn, axis=(0,2,3), keepdims=True) # (1,C,1,1)
        sum_dxn_xn = xp.sum(dxn * self.xn, axis=(0,2,3), keepdims=True) # (1,C,1,1)
        
        dx = (self.gamma.reshape(1,C,1,1) / self.std) * \
             (dxn - (sum_dxn / M) - self.xn * (sum_dxn_xn / M))
             
        return dx

class GlobalAveragePooling:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x):
        self.cache = x.shape 
        out = xp.mean(x, axis=(2, 3)) # (N, C)
        return out

    def backward(self, dout):
        N, C, H, W = self.cache
        dx_avg = dout.reshape(N, C, 1, 1) / (H * W) 
        dx = xp.tile(dx_avg, (1, 1, H, W))
        return dx