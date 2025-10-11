#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_worker.py
边缘节点脚本（树莓派）：
 - 周期轮询 master 的 /params?node=...&round=...
 - 收到 params（包含 base64 的 A_part npz、y、z、v、rho、lam）后执行一次 ADMM 更新（x,z,v）
 - 将 z_new, v_new, x 以 npz 上传到 master 的 /submit（并包含 meta）
使用：
    python3 edge_worker.py --master http://<master_ip>:8000 --node edge_0 --rounds 10 --token changeme_token
"""

import os
import io
import time
import json
import base64
import hashlib
import argparse
from typing import Optional

import numpy as np
import requests

# -----------------------
# 默认配置（可由命令行覆盖）
# -----------------------
MASTER_URL = "http://192.168.201.154:8000"
NODE_ID = "edge_0"
POLL_INTERVAL = 2.0
AUTH_TOKEN = "changeme_token"

SAVE_DIR = os.path.expanduser("~/received_parts")
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------
# 辅助（序列化/ADMM）
# -----------------------
def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

def npz_bytes_from_array_dict(d: dict) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, **d)
    return bio.getvalue()

def b64_to_bytes(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)

def update_z(x: np.ndarray, v: np.ndarray, lam: float, rho: float) -> np.ndarray:
    tau = lam / rho
    return soft_threshold(x + v, tau)

def update_v(v: np.ndarray, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    return v + (x - z)

def update_x_direct(A: np.ndarray, y: np.ndarray, z: np.ndarray, v: np.ndarray, rho: float) -> np.ndarray:
    AtA = A.T @ A
    Aty = A.T @ y
    n = AtA.shape[0]
    M = AtA + rho * np.eye(n, dtype=float)
    b = Aty + rho * (z - v)
    x = np.linalg.solve(M, b)
    return x

# -----------------------
# 与 master 通信
# -----------------------
def fetch_params_for_round(round_idx: int, timeout: float = 5.0) -> Optional[dict]:
    url = f"{MASTER_URL}/params?node={NODE_ID}&round={round_idx}"
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("ok"):
            return data.get("params")
        else:
            print("[EDGE] master 返回非 ok:", data)
            return None
    except requests.RequestException as e:
        print("[EDGE] fetch params failed:", e)
        return None

def submit_result_to_master(result_dict: dict, round_idx: int, part_id: str):
    url = f"{MASTER_URL}/submit"
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    # result_dict 是 {'z': z_arr, 'v': v_arr, 'x': x_arr}
    data_bytes = npz_bytes_from_array_dict(result_dict)
    meta = {"node": NODE_ID, "round": round_idx, "part_id": part_id, "sha256": sha256_bytes(data_bytes)}
    files = {"file": ("result.npz", data_bytes, "application/octet-stream")}
    payload = {"meta": json.dumps(meta)}
    try:
        r = requests.post(url, headers=headers, files=files, data=payload, timeout=30)
        r.raise_for_status()
        print(f"[EDGE] 上报成功，master 响应: {r.text}")
    except requests.RequestException as e:
        print("[EDGE] 上报失败:", e)

# -----------------------
# 主循环：轮询拉取参数 -> ADMM 单步 -> 回传
# -----------------------
def run_event_loop(max_rounds: int = 10):
    round_idx = 0
    while round_idx < max_rounds:
        print(f"[EDGE] 轮次 {round_idx}: 拉取 master 参数 ...")
        params = fetch_params_for_round(round_idx)
        if params is None:
            time.sleep(POLL_INTERVAL)
            continue

        print(f"[EDGE] 收到参数（示例 keys）: {list(params.keys())}")
        # 解码 A_part
        try:
            a_b64 = params.get("A_npz_b64")
            if a_b64 is None:
                print("[EDGE] params 中没有 A_npz_b64，跳过")
                time.sleep(POLL_INTERVAL)
                continue
            a_bytes = b64_to_bytes(a_b64)
            arr_dict = np.load(io.BytesIO(a_bytes), allow_pickle=False)
            A_part = arr_dict["A"]
        except Exception as e:
            print("[EDGE] 无法解析 A_part:", e)
            time.sleep(POLL_INTERVAL)
            continue

        # 取出 y, z, v, rho, lam
        y = np.asarray(params.get("y", []), dtype=float)
        z = np.asarray(params.get("z", []), dtype=float)
        v = np.asarray(params.get("v", []), dtype=float)
        rho = float(params.get("rho", 1.0))
        lam = float(params.get("lam", 0.1))
        part_id = params.get("part_id", f"{NODE_ID}_part")

        # 确保向量形状匹配：z/v 的长度应该等于 A_part 的列数
        if z.shape[0] != A_part.shape[1]:
            z = np.zeros(A_part.shape[1], dtype=float)
        if v.shape[0] != A_part.shape[1]:
            v = np.zeros(A_part.shape[1], dtype=float)

        # === ADMM 单步更新（边缘） ===
        x_new = update_x_direct(A_part, y, z, v, rho)
        z_new = update_z(x_new, v, lam, rho)
        v_new = update_v(v, x_new, z_new)

        # 保存本地（便于调试/持久化）
        save_path = os.path.join(SAVE_DIR, f"{NODE_ID}_r{round_idx}_{part_id}.npz")
        np.savez_compressed(save_path, x=x_new, z=z_new, v=v_new)
        print(f"[EDGE] 本地保存结果：{save_path}")

        # 上传到 master
        submit_result_to_master({"z": z_new, "v": v_new, "x": x_new}, round_idx, part_id)

        # 下一轮
        round_idx += 1
        time.sleep(0.5)

    print("[EDGE] 达到最大轮次，退出。")

# -----------------------
# CLI 入口
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=str, default=MASTER_URL)
    parser.add_argument("--node", type=str, default=NODE_ID)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--token", type=str, default=AUTH_TOKEN)
    args = parser.parse_args()

    MASTER_URL = args.master
    NODE_ID = args.node
    AUTH_TOKEN = args.token

    print(f"[EDGE] 启动：node={NODE_ID}, master={MASTER_URL}, token={AUTH_TOKEN}")
    run_event_loop(max_rounds=args.rounds)
