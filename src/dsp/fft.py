import math
from typing import Iterable

import numpy as np


def _next_pow_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _bit_reverse_indices(n: int) -> np.ndarray:
    bits = int(math.log2(n))
    indices = np.arange(n, dtype=np.uint32)
    rev = np.zeros(n, dtype=np.uint32)
    for i in range(n):
        b = indices[i]
        r = 0
        for _ in range(bits):
            r = (r << 1) | (b & 1)
            b >>= 1
        rev[i] = r
    return rev


def fft(x: Iterable[complex], n: int | None = None) -> np.ndarray:
    """Compute FFT using radix-2 Cooley-Tukey (no numpy.fft)."""
    x = np.asarray(list(x), dtype=np.complex128)
    if n is None:
        n = x.shape[0]
    if n < x.shape[0]:
        x = x[:n]
    elif n > x.shape[0]:
        pad = np.zeros(n - x.shape[0], dtype=np.complex128)
        x = np.concatenate([x, pad])

    n_fft = _next_pow_two(n)
    if n_fft != n:
        pad = np.zeros(n_fft - n, dtype=np.complex128)
        x = np.concatenate([x, pad])
        n = n_fft

    if n == 1:
        return x

    rev = _bit_reverse_indices(n)
    x = x[rev]

    m = 2
    while m <= n:
        half = m // 2
        angle = -2j * math.pi / m
        w_m = np.exp(np.arange(half) * angle)
        for k in range(0, n, m):
            t = w_m * x[k + half : k + m]
            u = x[k : k + half].copy()
            x[k : k + half] = u + t
            x[k + half : k + m] = u - t
        m *= 2
    return x


def ifft(x: Iterable[complex], n: int | None = None) -> np.ndarray:
    """Compute inverse FFT using conjugation trick."""
    x = np.asarray(list(x), dtype=np.complex128)
    if n is None:
        n = x.shape[0]
    y = fft(np.conjugate(x), n=n)
    return np.conjugate(y) / y.shape[0]


def rfft(x: Iterable[float], n: int | None = None) -> np.ndarray:
    """Real FFT wrapper based on complex FFT."""
    x = np.asarray(list(x), dtype=np.float64)
    y = fft(x, n=n)
    return y[: y.shape[0] // 2 + 1]
