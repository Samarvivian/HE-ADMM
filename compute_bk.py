from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def _load_ak_from_npz(path: str, npz_key: Optional[str] = None) -> np.ndarray:
    """
    Load matrix A_k from an .npz file. If npz_key is provided, use it; otherwise
    prefer common key 'arr', else fall back to the first available key.

    This function is optimized to work with data received from receive_ak function.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            if npz_key is not None:
                if npz_key not in data:
                    raise KeyError(f"Key '{npz_key}' not found in {path}")
                return data[npz_key]

            # 优先尝试常见的键名，与receive_ak函数保持一致
            preferred_keys = ['arr', 'Ak', 'A_k', 'matrix', 'data']
            for key in preferred_keys:
                if key in data:
                    return data[key]

            # Fallback to the first key deterministically
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"No arrays found in npz file: {path}")
            return data[keys[0]]
    except Exception as e:
        raise RuntimeError(f"Failed to load matrix from {path}: {e}")


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

    This function is optimized to work with data received from receive_ak function.

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
    print(f"[COMPUTE_BK] 开始计算Bk，输入类型: {type(Ak)}, rho: {rho}")

    # 处理输入数据
    if isinstance(Ak, str):
        print(f"[COMPUTE_BK] 从文件加载Ak: {Ak}, npz_key: {npz_key}")
        Ak = _load_ak_from_npz(Ak, npz_key=npz_key)
        print(f"[COMPUTE_BK] 成功加载Ak，形状: {Ak.shape}, 数据类型: {Ak.dtype}")
    elif isinstance(Ak, np.ndarray):
        print(f"[COMPUTE_BK] 使用numpy数组Ak，形状: {Ak.shape}, 数据类型: {Ak.dtype}")
    else:
        print(f"[COMPUTE_BK] 转换输入为numpy数组")
        Ak = np.asarray(Ak)

    # 验证输入
    if Ak.ndim != 2:
        raise ValueError(f"Ak must be a 2D array, got shape: {Ak.shape}")

    if not np.issubdtype(Ak.dtype, np.floating):
        print(f"[COMPUTE_BK] 转换数据类型从 {Ak.dtype} 到 float64")
        Ak = Ak.astype(np.float64, copy=False)

    if not np.isscalar(rho):
        raise ValueError("rho must be a scalar")
    rho = float(rho)
    if rho < 0:
        raise ValueError("rho must be non-negative")

    print(f"[COMPUTE_BK] 开始计算Gram矩阵 A_k^T A_k，Ak形状: {Ak.shape}")
    num_cols = Ak.shape[1]
    gram = Ak.T @ Ak
    print(f"[COMPUTE_BK] Gram矩阵形状: {gram.shape}")

    if rho > 0.0:
        print(f"[COMPUTE_BK] 添加正则化项 rho*I，rho = {rho}")
        gram = gram + rho * np.eye(num_cols, dtype=gram.dtype)

    # 尝试直接逆矩阵，如果失败则使用伪逆
    print(f"[COMPUTE_BK] 开始矩阵求逆...")
    try:
        Bk = np.linalg.inv(gram)
        print("✓ 成功计算 B_k = (A_k^T A_k + rho I)^{-1} (使用直接逆矩阵)")
    except np.linalg.LinAlgError as e:
        print(f"[COMPUTE_BK] 直接逆矩阵失败: {e}")
        print("[COMPUTE_BK] 尝试使用伪逆矩阵...")
        Bk = np.linalg.pinv(gram)
        print("✓ 成功计算 B_k = (A_k^T A_k + rho I)^{-1} (使用伪逆矩阵)")

    print(f"[COMPUTE_BK] 计算完成，Bk形状: {Bk.shape}, 数据类型: {Bk.dtype}")
    return Bk


__all__ = ["compute_Bk"]



