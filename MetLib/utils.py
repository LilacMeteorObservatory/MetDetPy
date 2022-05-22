import numpy as np

def m3func(image_stack, skipping=1):
    """M3 for Max Minus Median.
    Args:
        image_stack (ndarray)
    """
    sort_stack = np.sort(image_stack[::skipping], axis=0)
    return sort_stack[-1] - sort_stack[len(sort_stack) // 2]


#####################################################
##WARNING: The Following Functions Are Deprecatred.##
#####################################################

def GammaCorrection(src, gamma):
    """deprecated."""
    return np.array((src / 255.)**(1 / gamma) * 255, dtype=np.uint8)


#@numba.jit(nopython=True, fastmath=True, parallel=True)
def sanaas(stack: np.array, des: np.array, boolmap, L, H, W):
    '''
    ================================DEPRECATED========================
    stack [list[np.array]],
    new_matrix H*W
    des[list[np.array]] L*H*W stack[des]是真实升序序列。des是下标序列。
    已实现可借助numba的加速版本；但令人悲伤的仍然是其运行速度。因此废弃。
    ================================DEPRECATED========================
    '''
    # 双向冒泡
    # stack and des : L*(HW)
    for k in np.arange(0, L - 1, 1):
        # boolmap : (HW,) (bool)
        for i in np.arange(H * W):
            boolmap[i] = stack[des[k, i], i] > stack[des[k + 1, i], i]
        des[k, np.where(boolmap)[0]], des[k + 1, np.where(boolmap)[0]] = des[
            k + 1, np.where(boolmap)[0]], des[k, np.where(boolmap)[0]]

    for k in np.arange(L - 2, -1, -1):
        # boolmap : (HW,) (bool)
        for i in np.arange(H * W):
            boolmap[i] = stack[des[k, i], i] > stack[des[k + 1, i], i]
        for position in np.where(boolmap)[0]:
            des[k, position], des[k + 1, position] = des[k + 1, position], des[
                k, position]
    return stack, des

def series_keeping(sort_stack, frame, window_size, boolmap, L, H, W):
    sort_stack = np.concatenate((sort_stack[1:], sort_stack[:1]), axis=0)
    # Reshape为二维
    window_stack = np.reshape(window_stack, (L, H * W))
    sort_stack = np.reshape(sort_stack, (L, H * W))
    # numba加速的双向冒泡
    window_stack, sort_stack = sanaas(window_stack, sort_stack, boolmap, L,
                                      H, W)
    # 计算max-median
    diff_img = np.reshape(window_stack[sort_stack[-1],
                                       np.where(sort_stack[-1] >= 0)[0]],
                          (H, W))
    # 形状还原
    window_stack = np.reshape(window_stack, (L, H, W))
    sort_stack = np.reshape(sort_stack, (L, H, W))

    #update and calculate
    window_stack.append(frame)
    if len(window_stack) > window_size:
        window_stack.pop(0)
    diff_img = np.max(window_stack, axis=0) - np.median(window_stack, axis=0)
    return diff_img