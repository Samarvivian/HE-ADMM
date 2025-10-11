#!/usr/bin/env python3
"""
测试脚本：运行 compute_Bk 函数
"""

import numpy as np
from compute_bk import compute_Bk


def test_compute_bk():
    """测试 compute_Bk 函数的不同情况"""
    
    print("=" * 60)
    print("测试 compute_Bk 函数")
    print("=" * 60)
    
    # 测试1: 使用随机矩阵，rho > 0 (正常情况)
    print("\n【测试1】随机矩阵，rho = 0.1")
    print("-" * 40)
    np.random.seed(42)  # 设置随机种子以便复现
    Ak1 = np.random.randn(5, 3)
    rho1 = 0.1
    
    print(f"输入矩阵 A_k 形状: {Ak1.shape}")
    print(f"正则化参数 rho: {rho1}")
    print(f"A_k 前几行:\n{Ak1[:2]}")
    
    Bk1 = compute_Bk(Ak1, rho1)
    print(f"输出矩阵 B_k 形状: {Bk1.shape}")
    print(f"B_k 前几行:\n{Bk1[:2]}")
    
    # 验证结果
    gram1 = Ak1.T @ Ak1 + rho1 * np.eye(Ak1.shape[1])
    expected_identity = gram1 @ Bk1
    print(f"验证: ||(A_k^T A_k + rho I) B_k - I||_F = {np.linalg.norm(expected_identity - np.eye(Ak1.shape[1])):.2e}")
    
    # 测试2: 使用随机矩阵，rho = 0 (可能奇异)
    print("\n【测试2】随机矩阵，rho = 0")
    print("-" * 40)
    Ak2 = np.random.randn(4, 4)
    rho2 = 0.0
    
    print(f"输入矩阵 A_k 形状: {Ak2.shape}")
    print(f"正则化参数 rho: {rho2}")
    print(f"A_k 前几行:\n{Ak2[:2]}")
    
    Bk2 = compute_Bk(Ak2, rho2)
    print(f"输出矩阵 B_k 形状: {Bk2.shape}")
    print(f"B_k 前几行:\n{Bk2[:2]}")
    
    # 测试3: 使用奇异矩阵 (行数 < 列数)
    print("\n【测试3】奇异矩阵 (3x5)")
    print("-" * 40)
    Ak3 = np.random.randn(3, 5)
    rho3 = 0.01
    
    print(f"输入矩阵 A_k 形状: {Ak3.shape}")
    print(f"正则化参数 rho: {rho3}")
    print(f"A_k:\n{Ak3}")
    
    Bk3 = compute_Bk(Ak3, rho3)
    print(f"输出矩阵 B_k 形状: {Bk3.shape}")
    print(f"B_k 前几行:\n{Bk3[:2]}")
    
    # 测试4: 使用已知的简单矩阵
    print("\n【测试4】简单测试矩阵")
    print("-" * 40)
    Ak4 = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    rho4 = 0.1
    
    print(f"输入矩阵 A_k:\n{Ak4}")
    print(f"正则化参数 rho: {rho4}")
    
    Bk4 = compute_Bk(Ak4, rho4)
    print(f"输出矩阵 B_k:\n{Bk4}")
    
    # 验证结果
    gram4 = Ak4.T @ Ak4 + rho4 * np.eye(Ak4.shape[1])
    expected_identity4 = gram4 @ Bk4
    print(f"验证: ||(A_k^T A_k + rho I) B_k - I||_F = {np.linalg.norm(expected_identity4 - np.eye(Ak4.shape[1])):.2e}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_compute_bk()