import numpy as np
from typing import Tuple
import math


class QuantizationFunctions:
    """
    量化函数实现 - 基于论文公式(14a-d)
    """

    def __init__(self, delta: float = 1000.0):
        self.delta = delta

    def gamma1_quantize(self, u: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
        """
        Γ₁量化函数 - 公式(14a)
        Γ₁(u) = ⌊Δ²(u - u_min·1_Nk) / (u_max - u_min)²⌉

        用于加密 B_k(A_k^T y)
        """
        if u_max <= u_min:
            return np.zeros_like(u, dtype=np.int64)

        denominator = (u_max - u_min) ** 2
        scale = self.delta ** 2 / denominator

        # 确保向量操作
        if u.ndim == 1:
            u_min_vector = u_min * np.ones_like(u)
        else:
            u_min_vector = u_min * np.ones_like(u)

        quantized = np.floor(scale * (u - u_min_vector))
        return np.clip(quantized, 0, None).astype(np.int64)

    def gamma2_quantize(self, u: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
        """
        Γ₂量化函数 - 公式(14b, 14c, 14d)
        Γ₂(u) = ⌊Δ(u - u_min·1_Nk) / (u_max - u_min)⌉

        用于加密 z_k, v_k, B_k
        """
        if u_max <= u_min:
            return np.zeros_like(u, dtype=np.int64)

        scale = self.delta / (u_max - u_min)

        if u.ndim == 1:
            u_min_vector = u_min * np.ones_like(u)
        else:
            u_min_vector = u_min * np.ones_like(u)

        quantized = np.floor(scale * (u - u_min_vector))
        return np.clip(quantized, 0, None).astype(np.int64)

    def inverse_quantize_gamma1(self, u_quant: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
        """Γ₁反量化"""
        if u_max <= u_min:
            return np.full_like(u_quant, u_min, dtype=float)
        scale = (u_max - u_min) ** 2 / self.delta ** 2
        return u_quant * scale + u_min

    def inverse_quantize_gamma2(self, u_quant: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
        """Γ₂反量化"""
        if u_max <= u_min:
            return np.full_like(u_quant, u_min, dtype=float)
        scale = (u_max - u_min) / self.delta
        return u_quant * scale + u_min


def test_quantization_precision():
    """测试量化与反量化的精度损失"""
    print("量化精度测试")
    print("=" * 50)

    quantizer = QuantizationFunctions(delta=1000.0)

    # 测试数据
    np.random.seed(42)
    original_data = np.random.randn(10) * 5 + 2  # 均值为2，标准差为5的正态分布

    # 计算量化范围
    u_min, u_max = np.min(original_data), np.max(original_data)

    print(f"原始数据范围: [{u_min:.4f}, {u_max:.4f}]")
    print(f"原始数据: {original_data}")

    # Γ₁量化测试
    gamma1_quantized = quantizer.gamma1_quantize(original_data, u_min, u_max)
    gamma1_recovered = quantizer.inverse_quantize_gamma1(gamma1_quantized, u_min, u_max)

    gamma1_error = np.linalg.norm(original_data - gamma1_recovered)
    gamma1_relative_error = gamma1_error / np.linalg.norm(original_data)

    print(f"\nΓ₁量化测试:")
    print(f"量化后: {gamma1_quantized}")
    print(f"恢复后: {gamma1_recovered}")
    print(f"绝对误差: {gamma1_error:.6f}")
    print(f"相对误差: {gamma1_relative_error:.6f}")

    # Γ₂量化测试
    gamma2_quantized = quantizer.gamma2_quantize(original_data, u_min, u_max)
    gamma2_recovered = quantizer.inverse_quantize_gamma2(gamma2_quantized, u_min, u_max)

    gamma2_error = np.linalg.norm(original_data - gamma2_recovered)
    gamma2_relative_error = gamma2_error / np.linalg.norm(original_data)

    print(f"\nΓ₂量化测试:")
    print(f"量化后: {gamma2_quantized}")
    print(f"恢复后: {gamma2_recovered}")
    print(f"绝对误差: {gamma2_error:.6f}")
    print(f"相对误差: {gamma2_relative_error:.6f}")

    # 测试不同Δ值的影响
    print(f"\n不同Δ值的精度影响:")
    delta_values = [100, 1000, 10000]
    for delta in delta_values:
        quantizer_test = QuantizationFunctions(delta=delta)
        quantized = quantizer_test.gamma1_quantize(original_data, u_min, u_max)
        recovered = quantizer_test.inverse_quantize_gamma1(quantized, u_min, u_max)
        error = np.linalg.norm(original_data - recovered)
        print(f"  Δ={delta}: 误差={error:.6f}")


if __name__ == "__main__":
    test_quantization_precision()