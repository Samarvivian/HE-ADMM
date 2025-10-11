from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def _load_ak_from_npz(path: str, npz_key: Optional[str] = None) -> np.ndarray:
    """
    Load matrix A_k from an .npz file. If npz_key is provided, use it; otherwise
    prefer common key 'arr', else fall back to the first available key.
    """
    with np.load(path) as data:
        if npz_key is not None:
            if npz_key not in data:
                raise KeyError(f"Key '{npz_key}' not found in {path}")
            return data[npz_key]

        if "arr" in data:
            return data["arr"]

        # Fallback to the first key deterministically
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"No arrays found in npz file: {path}")
        return data[keys[0]]


def compute_Bk(
    Ak: Union[np.ndarray, str],
    rho: float,
    npz_key: Optional[str] = None,
) -> np.ndarray:
    """
    Compute B_k = (A_k^T A_k + rho I)^{-1}.

    - If Ak is a NumPy array, compute directly.
    - If Ak is a path to an .npz file, load using the provided npz_key (or infer).
    - When rho == 0 and the Gram matrix is singular, fall back to Moore-Penrose pseudo-inverse.

    Parameters
    ----------
    Ak : np.ndarray | str
        Matrix A_k or path to an .npz containing it.
    rho : float
        Non-negative regularization parameter.
    npz_key : Optional[str]
        Optional key in the .npz file to load A_k from when Ak is a path.

    Returns
    -------
    np.ndarray
        The matrix B_k with shape (n, n), where n is the number of columns of A_k.
    """
    if isinstance(Ak, str):
        Ak = _load_ak_from_npz(Ak, npz_key=npz_key)

    if not isinstance(Ak, np.ndarray):
        Ak = np.asarray(Ak)

    if Ak.ndim != 2:
        raise ValueError("Ak must be a 2D array")

    if not np.issubdtype(Ak.dtype, np.floating):
        Ak = Ak.astype(np.float64, copy=False)

    if not np.isscalar(rho):
        raise ValueError("rho must be a scalar")
    rho = float(rho)
    if rho < 0:
        raise ValueError("rho must be non-negative")

    num_cols = Ak.shape[1]
    gram = Ak.T @ Ak

    if rho > 0.0:
        gram = gram + rho * np.eye(num_cols, dtype=gram.dtype)

    # Try direct inversion first when regularized; fall back to pseudo-inverse if needed
    try:
        Bk = np.linalg.inv(gram)
        print("✓ 成功计算 B_k = (A_k^T A_k + rho I)^{-1} (使用直接逆矩阵)")
    except np.linalg.LinAlgError:
        Bk = np.linalg.pinv(gram)
        print("✓ 成功计算 B_k = (A_k^T A_k + rho I)^{-1} (使用伪逆矩阵)")

    return Bk


__all__ = ["compute_Bk"]



