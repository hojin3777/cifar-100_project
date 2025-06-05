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

def smooth_curve(x):
    """손실 함수의 그래프를 매끄럽게 하기 위해 사용
    참고：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    # 이 함수는 시각화용이므로 NumPy로 유지하거나, 결과를 CPU로 옮겨 처리
    if isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def shuffle_dataset(x, t):
    """데이터셋을 뒤섞는다.
    Parameters
    ----------
    x : 훈련 데이터 (CuPy 배열)
    t : 정답 레이블 (CuPy 배열)
    Returns
    -------
    x, t : 뒤섞은 훈련 데이터와 정답 레이블 (CuPy 배열)
    """
    permutation = xp.random.permutation(x.shape[0])
    x = x[permutation] # CuPy는 고급 인덱싱을 NumPy와 유사하게 지원
    t = t[permutation]
    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return int((input_size + 2*pad - filter_size) / stride + 1)

def pool_output_size(input_size, pool_size, stride): # common.util.py에는 없지만, CustomCNN에서 사용하던 것
    return int(1 + (input_size - pool_size) / stride)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = conv_output_size(H, filter_h, stride, pad)
    out_w = conv_output_size(W, filter_w, stride, pad)

    # 패딩 적용: CuPy는 cp.pad를 사용
    img = xp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = conv_output_size(H, filter_h, stride, pad)
    out_w = conv_output_size(W, filter_w, stride, pad)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), dtype=col.dtype)
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            # img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :] # 원본
            # CuPy에서는 += 연산 시 주의가 필요할 수 있으나, 대부분 잘 동작합니다.
            # 또는 cp.add.at을 사용하는 방법도 고려할 수 있지만, 여기서는 직접 할당을 유지합니다.
            # img.scatter_add((slice(None), slice(None), slice(y,y_max,stride), slice(x,x_max,stride)), col[:,:,y,x,:,:]) # cupy.scatter_add
            # 아래는 NumPy 방식과 유사하게 CuPy에서 작동하는 코드입니다.
            # 슬라이싱을 사용하여 업데이트할 부분을 지정하고 값을 더합니다.
            # CuPy에서 += 연산은 일반적으로 잘 작동합니다.
            # 만약 문제가 발생한다면, cp.add.at을 고려할 수 있습니다.
            # img[:, :, y:y_max:stride, x:x_max:stride] = img[:, :, y:y_max:stride, x:x_max:stride] + col[:, :, y, x, :, :]
            # 더 간단하게는 다음과 같습니다.
            img_slice = img[:, :, y:y_max:stride, x:x_max:stride]
            img_slice += col[:, :, y, x, :, :]


    return img[:, :, pad:H + pad, pad:W + pad]