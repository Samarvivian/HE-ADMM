from __future__ import annotations

import numpy as np


def _uniform_affine_quantize(matrix: np.ndarray, num_bits: int = 8, symmetric: bool = True) -> tuple[np.ndarray, float]:
    """
    Uniformly quantize a floating matrix to signed integers with an affine scale.

    Returns (q_matrix, scale) such that matrix ≈ q_matrix * scale.

    Parameters
    ----------
    matrix : np.ndarray
        Floating-point input matrix.
    num_bits : int
        Bit-width for quantization (default 8 -> int8 range [-128, 127]).
    symmetric : bool
        If True, use symmetric range [-Qmax, Qmax]; else use two-sided asymmetric.
    """

    if num_bits < 2 or num_bits > 16:
        raise ValueError("num_bits must be between 2 and 16")

    qmax = (1 << (num_bits - 1)) - 1  # e.g., 127 for 8-bit
    qmin = -qmax if symmetric else -(1 << (num_bits - 1))

    abs_max = float(np.max(np.abs(matrix))) if matrix.size > 0 else 0.0
    if abs_max == 0.0:
        return np.zeros_like(matrix, dtype=np.int8 if num_bits <= 8 else np.int16), 1.0

    scale = abs_max / qmax
    q = np.round(matrix / scale)
    q = np.clip(q, qmin, qmax)

    dtype = np.int8 if num_bits <= 8 else np.int16
    return q.astype(dtype, copy=False), scale


def compute_Bk_rho(Bk: np.ndarray, rho: float, num_bits: int = 8) -> tuple[np.ndarray, float]:
    """
    Compute B_{k, rho} = rho * B_k and quantize it to \u0304B_k (B_bar_k).

    Returns (B_bar_k, scale) such that approximately:
        rho * B_k ≈ B_bar_k * scale

    Parameters
    ----------
    Bk : np.ndarray
        The matrix B_k with shape (n, n).
    rho : float
        Non-negative scalar multiplier.
    num_bits : int
        Bit-width for integer quantization (default 8 results in int8 output).

    Returns
    -------
    tuple[np.ndarray, float]
        Quantized integer matrix B_bar_k and the floating scale.
    """

    if not isinstance(Bk, np.ndarray):
        Bk = np.asarray(Bk)

    if Bk.ndim != 2:
        raise ValueError("Bk must be a 2D array")

    if not np.isscalar(rho):
        raise ValueError("rho must be a scalar")

    if rho < 0:
        raise ValueError("rho must be non-negative")

    if not np.issubdtype(Bk.dtype, np.floating):
        Bk = Bk.astype(np.float64, copy=False)

    scaled = float(rho) * Bk
    B_bar_k, scale = _uniform_affine_quantize(scaled, num_bits=num_bits, symmetric=True)
    return B_bar_k, scale


__all__ = ["compute_Bk_rho"]


