#!/usr/bin/env python3
"""
send_bk.py - 边缘节点发送Bk矩阵给主节点
"""

import requests
import json
import hashlib
import numpy as np
import io
import os
from typing import Union, Optional, Dict, Any
from datetime import datetime


def sha256_bytes(data: bytes) -> str:
    """计算字节数据的SHA256哈希值"""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def send_bk(
        Bk: Union[np.ndarray, str],
        master_node_url: str = "http://localhost:5001",
        endpoint: str = "/receive_bk",
        part_id: Optional[str] = None,
        timeout: int = 30,
        npz_key: Optional[str] = None,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    将计算好的Bk矩阵发送给主节点

    Parameters
    ----------
    Bk : np.ndarray | str
        Bk矩阵或包含Bk矩阵的npz文件路径
    master_node_url : str
        主节点的URL地址，默认为 "http://localhost:5001"
    endpoint : str
        主节点接收Bk的端点，默认为 "/receive_bk"
    part_id : Optional[str]
        部分ID，如果不提供则自动生成
    timeout : int
        请求超时时间（秒），默认30秒
    npz_key : Optional[str]
        如果Bk是文件路径，指定npz文件中的键名
    verbose : bool
        是否打印详细日志，默认True

    Returns
    -------
    Dict[str, Any]
        包含发送结果的字典，包含success、message等信息
    """

    if verbose:
        print(f"[SEND_BK] 开始发送Bk矩阵到主节点: {master_node_url}{endpoint}")

    try:
        # 处理输入数据
        if isinstance(Bk, str):
            if verbose:
                print(f"[SEND_BK] 从文件加载Bk: {Bk}")
            Bk = _load_bk_from_file(Bk, npz_key=npz_key, verbose=verbose)
        elif isinstance(Bk, np.ndarray):
            if verbose:
                print(f"[SEND_BK] 使用numpy数组Bk，形状: {Bk.shape}, 数据类型: {Bk.dtype}")
        else:
            if verbose:
                print(f"[SEND_BK] 转换输入为numpy数组")
            Bk = np.asarray(Bk)

        # 验证输入
        if Bk.ndim != 2:
            error_msg = f"Bk必须是2D数组，当前形状: {Bk.shape}"
            if verbose:
                print(f"[SEND_BK] 错误: {error_msg}")
            return {"success": False, "error": error_msg, "message": "输入验证失败"}

        # 确保数据类型为浮点数
        if not np.issubdtype(Bk.dtype, np.floating):
            if verbose:
                print(f"[SEND_BK] 转换数据类型从 {Bk.dtype} 到 float64")
            Bk = Bk.astype(np.float64, copy=False)

        # 生成part_id
        if part_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            part_id = f"bk_{timestamp}"

        if verbose:
            print(f"[SEND_BK] 使用part_id: {part_id}")
            print(f"[SEND_BK] Bk矩阵信息: 形状={Bk.shape}, 数据类型={Bk.dtype}")
            print(f"[SEND_BK] 矩阵统计: min={Bk.min():.6f}, max={Bk.max():.6f}, mean={Bk.mean():.6f}")

        # 将Bk矩阵保存为npz格式的字节流
        if verbose:
            print(f"[SEND_BK] 将Bk矩阵转换为npz格式...")

        bio = io.BytesIO()
        np.savez(bio, arr=Bk)
        file_bytes = bio.getvalue()
        bio.close()

        # 计算文件哈希
        file_sha256 = sha256_bytes(file_bytes)
        if verbose:
            print(f"[SEND_BK] 文件大小: {len(file_bytes)} 字节")
            print(f"[SEND_BK] 文件SHA256: {file_sha256}")

        # 准备元数据
        meta = {
            "part_id": part_id,
            "shape": list(Bk.shape),
            "dtype": str(Bk.dtype),
            "sha256": file_sha256,
            "timestamp": datetime.now().isoformat(),
            "matrix_type": "Bk",
            "description": "Computed Bk matrix from edge node"
        }

        if verbose:
            print(f"[SEND_BK] 准备发送数据...")
            print(f"[SEND_BK] 元数据: {json.dumps(meta, indent=2)}")

        # 准备请求数据
        files = {
            'file': ('bk_matrix.npz', io.BytesIO(file_bytes), 'application/octet-stream')
        }
        data = {
            'meta': json.dumps(meta)
        }

        # 发送请求
        if verbose:
            print(f"[SEND_BK] 发送POST请求到: {master_node_url}{endpoint}")

        response = requests.post(
            f"{master_node_url}{endpoint}",
            files=files,
            data=data,
            timeout=timeout
        )

        # 处理响应
        if verbose:
            print(f"[SEND_BK] 收到响应: 状态码={response.status_code}")

        if response.status_code == 200:
            try:
                result = response.json()
                if verbose:
                    print(f"[SEND_BK] 响应内容: {json.dumps(result, indent=2)}")

                if result.get("ok", False):
                    success_msg = f"✓ 成功发送Bk矩阵到主节点 (part_id: {part_id})"
                    if verbose:
                        print(success_msg)
                    return {
                        "success": True,
                        "message": success_msg,
                        "part_id": part_id,
                        "response": result,
                        "file_size": len(file_bytes),
                        "sha256": file_sha256
                    }
                else:
                    error_msg = f"主节点返回错误: {result.get('msg', '未知错误')}"
                    if verbose:
                        print(f"[SEND_BK] 错误: {error_msg}")
                    return {
                        "success": False,
                        "error": error_msg,
                        "response": result
                    }
            except json.JSONDecodeError:
                error_msg = f"无法解析主节点响应JSON: {response.text}"
                if verbose:
                    print(f"[SEND_BK] 错误: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "response_text": response.text
                }
        else:
            error_msg = f"HTTP请求失败: {response.status_code} - {response.text}"
            if verbose:
                print(f"[SEND_BK] 错误: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "status_code": response.status_code,
                "response_text": response.text
            }

    except requests.exceptions.Timeout:
        error_msg = f"请求超时 (超过{timeout}秒)"
        if verbose:
            print(f"[SEND_BK] 错误: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "message": "网络超时"
        }
    except requests.exceptions.ConnectionError as e:
        error_msg = f"连接错误: {str(e)}"
        if verbose:
            print(f"[SEND_BK] 错误: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "message": "无法连接到主节点"
        }
    except Exception as e:
        error_msg = f"发送过程中发生未知错误: {str(e)}"
        if verbose:
            print(f"[SEND_BK] 错误: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "message": "发送失败"
        }
    finally:
        # 清理文件对象
        if 'files' in locals():
            for file_obj in files.values():
                if hasattr(file_obj[1], 'close'):
                    file_obj[1].close()


def _load_bk_from_file(file_path: str, npz_key: Optional[str] = None, verbose: bool = True) -> np.ndarray:
    """
    从文件中加载Bk矩阵

    Parameters
    ----------
    file_path : str
        文件路径
    npz_key : Optional[str]
        npz文件中的键名
    verbose : bool
        是否打印详细日志

    Returns
    -------
    np.ndarray
        加载的Bk矩阵
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if verbose:
        print(f"[LOAD_BK] 从文件加载: {file_path}")

    if file_path.endswith('.npz'):
        # 加载npz文件
        with np.load(file_path, allow_pickle=False) as data:
            if npz_key is not None:
                if npz_key not in data:
                    raise KeyError(f"键 '{npz_key}' 在文件中不存在")
                return data[npz_key]

            # 优先尝试常见的键名
            preferred_keys = ['arr', 'Bk', 'B_k', 'matrix', 'data']
            for key in preferred_keys:
                if key in data:
                    if verbose:
                        print(f"[LOAD_BK] 使用键名: {key}")
                    return data[key]

            # 使用第一个可用的键
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"npz文件中没有找到任何数组")
            if verbose:
                print(f"[LOAD_BK] 使用第一个键名: {keys[0]}")
            return data[keys[0]]

    elif file_path.endswith('.npy'):
        # 加载npy文件
        if verbose:
            print(f"[LOAD_BK] 加载npy文件")
        return np.load(file_path)

    else:
        raise ValueError(f"不支持的文件格式: {file_path}")


def test_send_bk():
    """测试send_bk函数"""
    print("=== 测试 send_bk 函数 ===")

    # 创建一个测试Bk矩阵
    Bk = np.random.rand(50, 50).astype(np.float64)
    print(f"测试Bk矩阵形状: {Bk.shape}")

    # 测试发送（这里使用一个不存在的URL来测试错误处理）
    result = send_bk(
        Bk=Bk,
        master_node_url="http://localhost:5001",  # 假设主节点在这个地址
        part_id="test_bk_001",
        verbose=True
    )

    print(f"\n发送结果: {result}")

    if result["success"]:
        print("✓ 测试成功")
    else:
        print("✗ 测试失败（这是预期的，因为主节点可能没有运行）")


if __name__ == "__main__":
    test_send_bk()