# edge_node.py
"""
完整的边缘节点（树莓派）接收 A_k -> 计算 B_k -> 量化保存并回传主节点
使用方法：
    修改 MASTER_URL、MASTER_TOKEN（如需）、ALLOWED_MASTER_IP（可选）
    安装依赖：pip3 install numpy flask requests matplotlib
    运行：python3 edge_node.py
"""

from __future__ import annotations
import os
import io
import json
import time
import hashlib
from typing import Any, Dict, Tuple, Optional

import numpy as np
import matplotlib
# 在无显示环境下使用 Agg 后端绘图
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file

import requests

# ========== 配置 ==========
# 主节点接收 B_k 的 URL（边缘将 B_k POST 回到该 URL）
MASTER_URL = os.environ.get("MASTER_URL", "http://192.168.201.154:8000/receive_result")
# 可选认证 token（如果主节点启用了 token 验证）
MASTER_TOKEN = os.environ.get("MASTER_TOKEN", None)
# 可选：只允许来自某个主节点 IP 的请求（提高安全）
ALLOWED_MASTER_IP = os.environ.get("ALLOWED_MASTER_IP", None)  # e.g. "192.168.201.10"

# Flask 监听端口（边缘接收主节点下发 A_k）
FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("EDGE_PORT", 5000))

# 本地文件保存目录（自动使用当前用户 home）
SAVE_DIR = os.path.join(os.path.expanduser("~"), "received_parts")
os.makedirs(SAVE_DIR, exist_ok=True)

# heatmap 最大显示尺寸（用于超大矩阵时下采样）
HEATMAP_MAX_SIZE = (400, 400)  # width x height in pixels

# ======================== 辅助函数 ========================
def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def npz_bytes_from_array(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, arr=arr)
    return bio.getvalue()

def save_bytes_to(path: str, b: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(b)

def quantize_linear(x: np.ndarray, delta: float, zmin: float, zmax: float, dtype=np.int32) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    线性量化：
      q = round((x - zmin) / delta)
    返回 (q, meta)
    meta 包含 delta, zmin, zmax, dtype
    """
    x = np.asarray(x, dtype=np.float64)
    q = np.round((x - zmin) / delta).astype(np.int64)
    qmin = np.iinfo(dtype).min
    qmax = np.iinfo(dtype).max
    q = np.clip(q, qmin, qmax).astype(dtype)
    meta = {"delta": float(delta), "zmin": float(zmin), "zmax": float(zmax), "dtype": str(dtype)}
    return q, meta

def inverse_quantize(q: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    delta = float(meta["delta"])
    zmin = float(meta["zmin"])
    return q.astype(np.float64) * delta + zmin

def generate_heatmap_and_save(arr: np.ndarray, out_png: str) -> Dict[str, Any]:
    """
    为矩阵生成 heatmap 并保存 PNG（离屏）。如果矩阵太大，会下采样。
    返回字典包含 png_path 和可能的采样信息。
    """
    info = {}
    # 如果数组很大，先采样到合适尺寸
    h, w = arr.shape
    # 计算目标像素大小（不超过 HEATMAP_MAX_SIZE）
    max_w, max_h = HEATMAP_MAX_SIZE
    scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
    if scale < 1.0:
        # 下采样：按比例取均匀采样点
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        # 使用 simple numpy slicing for uniform sampling
        xs = np.linspace(0, w - 1, new_w).astype(int)
        ys = np.linspace(0, h - 1, new_h).astype(int)
        arr_sampled = arr[np.ix_(ys, xs)]
        info["downsampled"] = True
        info["sample_shape"] = arr_sampled.shape
    else:
        arr_sampled = arr
        info["downsampled"] = False
        info["sample_shape"] = arr_sampled.shape

    # 绘图
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(arr_sampled, aspect="auto", interpolation="nearest")
    ax.set_title(f"shape={arr.shape}")
    ax.set_xlabel("cols")
    ax.set_ylabel("rows")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
    finally:
        plt.close(fig)
    info["png_path"] = out_png
    return info

def compute_Bk(Ak: np.ndarray, rho: float) -> np.ndarray:
    """
    计算 B_k = (A_k^T A_k + rho I)^{-1}
    优先用 np.linalg.inv（小维度），若 cholesky / inv 失败退回到伪逆
    """
    Ak = np.asarray(Ak, dtype=np.float64)
    AtA = Ak.T @ Ak
    n_k = AtA.shape[0]
    M = AtA + rho * np.eye(n_k, dtype=np.float64)
    try:
        # 试 cholesky 用以验证 M 是 SPD（可以提高数值稳定性）
        _L = np.linalg.cholesky(M)
        # 若 cholesky 成功，直接用 inv 也可以；若你希望高效可用 cho_solve（scipy）
        Bk = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # 若不可对称或非正定，退回 pseudoinverse
        Bk = np.linalg.pinv(M)
    return Bk

def send_Bk_to_master(Bk: np.ndarray, part_id: str) -> Tuple[bool, Optional[str]]:
    """
    将浮点 Bk 以 npz 上传到 MASTER_URL（multipart/form-data）
    返回 (ok, response_text)
    """
    data_bytes = npz_bytes_from_array(Bk)
    meta = {"part_id": f"{part_id}_Bk", "shape": Bk.shape, "dtype": str(Bk.dtype), "sha256": sha256_bytes(data_bytes)}
    files = {"file": ("Bk.npz", data_bytes, "application/octet-stream")}
    payload = {"meta": json.dumps(meta)}
    headers = {}
    if MASTER_TOKEN:
        headers["Authorization"] = f"Bearer {MASTER_TOKEN}"
    try:
        resp = requests.post(MASTER_URL, files=files, data=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return True, resp.text
    except Exception as e:
        return False, str(e)

# ======================== Flask 应用 ========================
app = Flask(__name__)

def _is_authorized_request(req) -> Tuple[bool, str]:
    """
    简单校验请求来源：优先按 IP 校验，其次按 Authorization token 校验。
    返回 (ok, reason)
    """
    client_ip = req.remote_addr
    # 如果设置了 ALLOWED_MASTER_IP，仅允许该 IP
    if ALLOWED_MASTER_IP:
        if client_ip != ALLOWED_MASTER_IP:
            return False, f"IP {client_ip} not allowed"
    # 如果设置了 MASTER_TOKEN，则需要 Authorization header
    if MASTER_TOKEN:
        auth = req.headers.get("Authorization", "")
        if auth != f"Bearer {MASTER_TOKEN}":
            return False, "Invalid or missing Authorization token"
    return True, "ok"

@app.route("/upload", methods=["POST"])
def upload():
    """
    接收主节点上传的 A_k（multipart/form-data）：
      - meta: JSON 字符串（包含 part_id, rho, quant_params）
      - file: Ak.npz（二进制）
    处理流程：
      1. 验证请求是否合法
      2. 保存原始 npz
      3. 解析 numpy 数组并打印 preview
      4. 计算 B_k
      5. 保存 B_k（.npy）以及量化后的 B_k*rho（.npy + meta.json）
      6. 将浮点 B_k 上传回主节点
      7. 生成 heatmap png 并返回保存路径与回传结果
    """
    # 1) 授权校验
    ok, reason = _is_authorized_request(request)
    if not ok:
        return jsonify({"ok": False, "msg": f"Unauthorized: {reason}"}), 403

    # 2) 解析 meta 和文件
    meta_json = request.form.get("meta")
    if not meta_json or "file" not in request.files:
        return jsonify({"ok": False, "msg": "missing meta or file"}), 400

    try:
        meta = json.loads(meta_json)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"invalid meta json: {e}"}), 400

    file_storage = request.files["file"]
    file_bytes = file_storage.read()
    part_id = meta.get("part_id", f"part_{int(time.time())}")
    rho = float(meta.get("rho", 1.0))
    quant_params = meta.get("quant_params", {"delta": 1e-5, "zmin": None, "zmax": None})

    # 3) 保存原始 npz 到磁盘
    raw_path = os.path.join(SAVE_DIR, f"{part_id}_Ak.npz")
    save_bytes_to(raw_path, file_bytes)

    # 校验 sha256（若 meta 提供）
    provided_sha = meta.get("sha256")
    computed_sha = sha256_bytes(file_bytes)
    sha_ok = (provided_sha is None) or (provided_sha == computed_sha)

    # 4) 解析 numpy 数组
    arr = None
    load_error = None
    try:
        bio = io.BytesIO(file_bytes)
        npz = np.load(bio, allow_pickle=False)
        if 'arr' in npz.files:
            arr = npz['arr']
        elif len(npz.files) > 0:
            arr = npz[npz.files[0]]
        else:
            load_error = "npz contains no arrays"
    except Exception as e:
        load_error = str(e)

    if arr is None:
        return jsonify({"ok": False, "msg": "cannot decode npz", "load_error": load_error, "raw_path": raw_path}), 400

    # 5) 确保为 2D 数组，打印预览与统计
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    print(f"[EDGE] Received part_id={part_id}, shape={arr.shape}, rho={rho}, sha_ok={sha_ok}")
    preview_rows = min(arr.shape[0], 6)
    preview_cols = min(arr.shape[1], 8)
    np.set_printoptions(precision=4, suppress=True)
    print("[EDGE] preview:")
    print(arr[:preview_rows, :preview_cols])
    print("[EDGE] stats: min=%g, max=%g, mean=%g" % (arr.min(), arr.max(), arr.mean()))

    # 6) 生成 heatmap（异步或同步都可以，这里同步生成）
    heatmap_png = os.path.join(SAVE_DIR, f"{part_id}_heatmap.png")
    try:
        heat_info = generate_heatmap_and_save(arr, heatmap_png)
    except Exception as e:
        heat_info = {"error": str(e)}

    # 7) 计算 B_k
    try:
        Bk = compute_Bk(arr, rho)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"compute Bk failed: {e}"}), 500

    # 8) 保存浮点 Bk
    Bk_path = os.path.join(SAVE_DIR, f"{part_id}_Bk.npy")
    np.save(Bk_path, Bk)

    # 9) 量化 Bk * rho 并保存
    Bk_rho = Bk * rho
    # 若 quant_params 中的 zmin/zmax 为 None，则使用实际 min/max（可用 percentile 替代）
    delta = float(quant_params.get("delta", 1e-5))
    zmin = quant_params.get("zmin")
    zmax = quant_params.get("zmax")
    if zmin is None:
        zmin = float(np.min(Bk_rho))
    if zmax is None:
        zmax = float(np.max(Bk_rho))
    q_int, q_meta = quantize_linear(Bk_rho, delta, zmin, zmax, dtype=np.int32)
    q_path = os.path.join(SAVE_DIR, f"{part_id}_Bk_rho_q.npy")
    np.save(q_path, q_int)
    q_meta_path = os.path.join(SAVE_DIR, f"{part_id}_Bk_rho_q_meta.json")
    with open(q_meta_path, "w") as fh:
        json.dump(q_meta, fh)

    # 10) 将浮点 Bk 以 npz 上传回主节点
    ok_send, resp_text = send_Bk_to_master(Bk, part_id)

    # 11) 返回详细信息给主节点（或调用者）
    result = {
        "ok": True,
        "part_id": part_id,
        "raw_path": raw_path,
        "sha_ok": sha_ok,
        "heatmap_png": heatmap_png if os.path.exists(heatmap_png) else None,
        "heat_info": heat_info,
        "Bk_path": Bk_path,
        "Bk_shape": list(Bk.shape),
        "Bk_rho_q_path": q_path,
        "q_meta_path": q_meta_path,
        "sent_to_master": ok_send,
        "master_resp": resp_text
    }
    print(f"[EDGE] process completed for {part_id}, sent_to_master={ok_send}")
    return jsonify(result)

# 可选接口：下载 heatmap / 文件
@app.route("/visual/<filename>", methods=["GET"])
def get_visual(filename):
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "msg": "not found"}), 404
    return send_file(path, mimetype="image/png")

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"ok": True, "save_dir": SAVE_DIR, "time": time.time()})

# ======================== 启动 ========================
if __name__ == "__main__":
    print(f"[EDGE] Starting edge server on {FLASK_HOST}:{FLASK_PORT}")
    print(f"[EDGE] SAVE_DIR = {SAVE_DIR}")
    print(f"[EDGE] MASTER_URL = {MASTER_URL}, MASTER_TOKEN set = {bool(MASTER_TOKEN)}")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
