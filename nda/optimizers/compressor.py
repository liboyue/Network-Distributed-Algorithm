try:
    import cupy as xp
except ImportError:
    import numpy as xp

# import numpy as np

def identity(x, *args, **kwargs):
    return x

# top_a
def top(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a >= dim:
        return x
    index_array = xp.argpartition(x, kth=a, axis=0)[a:]
    xp.put_along_axis(x, index_array, 0, axis=0)
    return x

# x = np.random.randint(0, 100, 24).reshape(6, 4)
# x
# top(x, 2)

# Random_a compressor, keep a values
def random(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a == dim:
        return x
    if x.ndim == 2:
        for i in range(x.shape[1]):
            zero_mask = xp.random.choice(dim, a, replace=False)
            x[zero_mask, i] = 0
    else:
        zero_mask = xp.random.choice(dim, a, replace=False)
        x[zero_mask] = 0
    return x


# gsgd_b
def gsgd(x, b):
    norm = xp.linalg.norm(x, axis=0)
    return norm / (2 ** (b - 1)) * xp.sign(x) * xp.floor(
                (2 ** (b - 1)) / norm * xp.abs(x) + xp.random.uniform(0, 1, x.shape)
            )


# random quantization 2-norm with level s
def random_quantization(x, s):
    dim = x.shape[0]
    xnorm = xp.linalg.norm(x)
    if s == 0 or xnorm == 0:
        return xp.zeros(dim, dtype=int)
    noise = xp.random.uniform(0, 1, dim)
    rounded = xp.floor(s * xp.abs(x) / xnorm + noise)
    compressed = (xnorm / s) * xp.sign(x) * rounded
    return compressed


# natural compression (power of 2 for each coordinate)
def natural_compression(x):
    dim = x.shape[0]
    logx = xp.ma.log2(xp.abs(x)).filled(-15)
    logx_floor = xp.floor(logx)
    noise = xp.random.uniform(0.0, 1.0, dim)
    leftx = xp.exp2(logx_floor)
    rounded = xp.floor(xp.ma.log2(xp.abs(x) + leftx * noise).filled(-15))
    compressed = xp.sign(x) * xp.exp2(rounded)
    return compressed
