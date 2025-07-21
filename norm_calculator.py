import numpy as np

def dual_norm(x, norm_type):
    if norm_type == 1:
        return np.linalg.norm(x, np.inf)
    elif norm_type == 2:
        return np.linalg.norm(x, 2)
    elif norm_type == np.inf:
        return np.linalg.norm(x, 1)
    else:
        raise ValueError("Unsupported norm")

def lmo(s, norm_type):
    if norm_type == 2:
        norm_s = np.linalg.norm(s)
        return -s / norm_s if norm_s != 0 else np.zeros_like(s)
    elif norm_type == 1:
        e = np.zeros_like(s)
        idx = np.argmax(np.abs(s))
        e[idx] = -np.sign(s[idx])
        return e
    elif norm_type == np.inf:
        return -np.sign(s)
    else:
        raise ValueError("Unsupported norm")
