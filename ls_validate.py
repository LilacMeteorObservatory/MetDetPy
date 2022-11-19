import numpy as np
import matplotlib.pyplot as plt


def least_square_fit(pts):
    """fit pts to the linear func Ax+By+C=0

    Args:
        pts (_type_): _description_

    Returns:
        tuple: parameters (A,B,C)
        float: average loss
    """
    pts = np.array(pts)
    avg_xy = np.mean(pts, axis=0)
    dpts = pts - avg_xy
    dxx, dyy = np.sum(dpts**2, axis=0)
    dxy = np.sum(dpts[:, 0] * dpts[:, 1])
    v, m = np.linalg.eig([[dxx, dxy], [dxy, dyy]])
    A, B = m[np.argmin(v)]
    B = -B
    C = -np.sum(np.array([A, B]) * avg_xy)
    loss = np.abs(np.einsum("a,ba->b", np.array([A, B]), pts) + C)
    return (A, B, C), loss


k = np.array([[539, 179], [552, 200], [545, 189], [542, 179], [553, 204],
              [540, 179], [556, 204], [545, 185], [550, 200]])

(A, B, C), loss = least_square_fit(k)
print(loss)
y = - (k[:, 0] * A + C) / B

plt.scatter(k[:, 0], k[:, 1])
plt.scatter(k[:, 0], y)
plt.show()
