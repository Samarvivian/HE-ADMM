from __future__ import annotations

import io
import json
import os
import hashlib
from typing import Optional
from datetime import datetime

import numpy as np
import requests
from flask import Flask, request, jsonify

from compute_bk import compute_Bk


app = Flask(__name__)


# Directory to persist received parts (for audit/debugging)
SAVE_DIR = os.path.expanduser("~/received_parts")
VISUAL_DIR = os.path.expanduser("~/matrix_visualization")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)


# Master endpoint to receive computed results; can be overridden via env
MASTER_URL = os.environ.get("MASTER_RESULT_URL", "http://192.168.201.154:8000/receive_result")


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _serialize_np_array(arr: np.ndarray) -> bytes:
    """Serialize array into compressed NPZ bytes (key 'arr')."""
    bio = io.BytesIO()
    np.savez_compressed(bio, arr=arr)
    return bio.getvalue()


def save_matrix_visualization(matrix: np.ndarray, part_id: str, matrix_type: str = "Ak") -> str:
    """Save matrix as a readable text file for debugging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{matrix_type}_{part_id}_{timestamp}.txt"
    filepath = os.path.join(VISUAL_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"矩阵类型: {matrix_type}\n")
        f.write(f"部分ID: {part_id}\n")
        f.write(f"时间戳: {timestamp}\n")
        f.write(f"形状: {matrix.shape}\n")
        f.write(f"数据类型: {matrix.dtype}\n")
        f.write(f"最小值: {np.min(matrix):.6f}\n")
        f.write(f"最大值: {np.max(matrix):.6f}\n")
        f.write(f"平均值: {np.mean(matrix):.6f}\n")
        f.write(f"标准差: {np.std(matrix):.6f}\n")
        f.write("-" * 50 + "\n")
        
        # 如果矩阵太大，只显示部分内容
        if matrix.size > 1000:
            f.write("矩阵太大，只显示前10x10部分:\n")
            display_matrix = matrix[:10, :10] if matrix.ndim == 2 else matrix[:100]
        else:
            f.write("完整矩阵内容:\n")
            display_matrix = matrix
        
        # 保存矩阵内容
        if matrix.ndim == 2:
            for i, row in enumerate(display_matrix):
                f.write(f"行 {i:3d}: {row}\n")
        else:
            f.write(f"一维数组: {display_matrix}\n")
        
        f.write("-" * 50 + "\n")
        f.write(f"文件保存路径: {filepath}\n")
    
    print(f"[EDGE] 矩阵可视化已保存: {filepath}")
    return filepath


def send_result_to_master(result_array: np.ndarray, part_id: str) -> requests.Response:
    data_bytes = _serialize_np_array(result_array)
    meta = {
        "part_id": part_id,
        "shape": tuple(result_array.shape),
        "dtype": str(result_array.dtype),
        "sha256": sha256_bytes(data_bytes),
    }

    files = {"file": ("result.npz", data_bytes, "application/octet-stream")}
    payload = {"meta": json.dumps(meta, ensure_ascii=False)}

    print(f"[EDGE] 回传 {part_id} 结果到主节点: {MASTER_URL}")
    resp = requests.post(MASTER_URL, files=files, data=payload, timeout=60)
    print(f"[EDGE] 主节点响应: {resp.status_code}, 内容前120字: {resp.text[:120]}")
    return resp


@app.route("/upload", methods=["POST"])
def upload_matrix():
    # Validate multipart form
    meta_json = request.form.get("meta")
    if not meta_json or "file" not in request.files:
        return jsonify({"ok": False, "msg": "missing meta or file"}), 400

    try:
        meta = json.loads(meta_json)
    except Exception:
        return jsonify({"ok": False, "msg": "invalid meta json"}), 400

    file_obj = request.files["file"]
    part_id = str(meta.get("part_id", "unknown_part"))

    # Persist raw upload
    save_path = os.path.join(SAVE_DIR, f"{part_id}.npz")
    file_bytes = file_obj.read()
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    print(f"[EDGE] 已接收文件并保存: {save_path} ({len(file_bytes)} bytes)")

    # Extract optional rho and npz key from meta
    rho: float = float(meta.get("rho", 1.0))
    npz_key: Optional[str] = meta.get("npz_key")

    # Load A_k and compute B_k
    try:
        # If a specific key is provided, use in-memory load to avoid re-writing file
        if npz_key is not None:
            with np.load(io.BytesIO(file_bytes)) as data:
                if npz_key not in data:
                    return jsonify({"ok": False, "msg": f"key '{npz_key}' not in npz"}), 400
                Ak = data[npz_key]
        else:
            # Default key from typical sender is 'arr'; fall back handled by compute_bk loader if using path
            # Here we read via numpy to avoid re-IO after saving
            with np.load(io.BytesIO(file_bytes)) as data:
                if "arr" in data:
                    Ak = data["arr"]
                else:
                    # Fallback: rely on the saved path and compute_bk's key inference
                    Ak = save_path

        # 保存Ak矩阵的可视化文件用于调试
        if isinstance(Ak, np.ndarray):
            ak_viz_path = save_matrix_visualization(Ak, part_id, "Ak")
            print(f"[EDGE] Ak矩阵可视化已保存: {ak_viz_path}")
        else:
            print(f"[EDGE] Ak是文件路径，跳过可视化: {Ak}")

        Bk = compute_Bk(Ak, rho=rho, npz_key=npz_key if isinstance(Ak, str) else None)
        
        # 保存Bk矩阵的可视化文件用于调试
        if isinstance(Bk, np.ndarray):
            bk_viz_path = save_matrix_visualization(Bk, part_id, "Bk")
            print(f"[EDGE] Bk矩阵可视化已保存: {bk_viz_path}")
        else:
            print(f"[EDGE] Bk不是numpy数组，跳过可视化: {type(Bk)}")
    except Exception as e:
        print(f"[EDGE][ERROR] 计算 Bk 失败: {e}")
        return jsonify({"ok": False, "msg": f"compute_Bk failed: {e}"}), 500

    try:
        send_result_to_master(Bk, part_id)
    except Exception as e:
        print(f"[EDGE][ERROR] 回传结果失败: {e}")
        return jsonify({"ok": False, "msg": f"send result failed: {e}"}), 502

    return jsonify({"ok": True, "msg": "Bk computed and sent"})


if __name__ == "__main__":
    host = os.environ.get("EDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("EDGE_PORT", "5000"))
    print(f"[EDGE] 启动服务: http://{host}:{port}  回传地址: {MASTER_URL}")
    app.run(host=host, port=port)


