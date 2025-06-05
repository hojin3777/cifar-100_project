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

def identity_function(x):
    return x

def step_function(x):
    return xp.array(x > 0, dtype=xp.int32)

def sigmoid(x):
    # CuPy의 exp 함수는 매우 큰 음수 값에 대해 0을 반환하여 안정적입니다.
    return 1 / (1 + xp.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    return xp.maximum(0, x)

def relu_grad(x):
    grad = xp.zeros_like(x)
    grad[x >= 0] = 1
    return grad

def softmax(x):
    # x가 1차원 벡터일 경우를 처리 (예: (N,))
    if x.ndim == 1:
        x = x - xp.max(x) # 오버플로 대책
        return xp.exp(x) / xp.sum(xp.exp(x))
    # x가 2차원 이상의 배열일 경우 (예: (N, C))
    x = x - xp.max(x, axis=-1, keepdims=True) # 오버플로 대책
    return xp.exp(x) / xp.sum(xp.exp(x), axis=-1, keepdims=True)

def mean_squared_error(y, t):
    return 0.5 * xp.sum((y - t)**2)

def cross_entropy_error(y, t):
    if y.ndim == 1: # 단일 샘플 처리
        y = y.reshape(1, y.shape[0])
        if t.ndim == 0: # t가 스칼라 정수
            t = xp.asarray([t]) 
        elif t.ndim == 1 and t.size != y.shape[1]: # t가 (1,) 형태의 정수 레이블
             pass
        else: # t가 (C,) 형태의 원-핫/부드러운 레이블
            t = t.reshape(1, t.shape[0])

    batch_size = y.shape[0]

    # t가 (N,C) 형태의 원-핫 또는 부드러운 레이블일 경우
    if t.ndim == 2 and t.shape == y.shape:
        # -sum(t_k * log(y_k)) 계산
        return -xp.sum(t * xp.log(y + 1e-7)) / batch_size
    # t가 (N,) 또는 (N,1) 형태의 정수 레이블일 경우 (Label Smoothing이 적용되지 않은 경우의 경로)
    elif t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1):
        t_int = t.astype(xp.int32).ravel() # (N,) 형태로 변환
        log_y = xp.log(y[xp.arange(batch_size), t_int] + 1e-7)
        return -xp.sum(log_y) / batch_size
    else:
        raise ValueError(f"Incompatible shapes for y({y.shape}) and t({t.shape}) in cross_entropy_error")

def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)