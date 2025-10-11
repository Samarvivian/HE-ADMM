# edge_server_visual.py
# 边缘节点接收 A_k，并打印 + 保存为文件 + 生成 heatmap png (离屏)
from flask import Flask, request, jsonify, send_file
import os
import io
import json
import hashlib
import numpy as np
import matplotlib
# 使用非交互后端，以便在无显示器环境下绘图
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)

# 保存目录（自动使用当前运行用户的 home）
SAVE_DIR = os.path.join(os.path.expanduser("~"), "received_parts")
os.makedirs(SAVE_DIR, exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _save_matrix_and_visual(arr: np.ndarray, base_name: str) -> dict:
    """
    保存矩阵为 .npy 和 .csv，并生成热力图 png。
    返回保存路径信息。
    """
    info = {}
    # 保存为 .npy
    npy_path = os.path.join(SAVE_DIR, f"{base_name}.npy")
    np.save(npy_path, arr)
    info['npy_path'] = npy_path

    # 保存为 csv（便于用 Excel 打开）
    csv_path = os.path.join(SAVE_DIR, f"{base_name}.csv")
    try:
        np.savetxt(csv_path, arr, delimiter=",", fmt="%g")
        info['csv_path'] = csv_path
    except Exception as e:
        info['csv_error'] = str(e)

    # 生成 heatmap png
    png_path = os.path.join(SAVE_DIR, f"{base_name}_heatmap.png")
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        # 若矩阵非常大，限制绘制尺寸以避免内存占用
        # 使用 imshow 绘制热力图
        im = ax.imshow(arr, aspect='auto', interpolation='nearest')
        ax.set_title(f"{base_name} shape={arr.shape}")
        ax.set_xlabel("cols")
        ax.set_ylabel("rows")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        info['png_path'] = png_path
    except Exception as e:
        info['png_error'] = str(e)

    return info

@app.route("/upload", methods=["POST"])
def upload_matrix():
    """
    接收主节点发送的 A_k（.npz）。处理流程：
      - 读取 meta（JSON）和文件 bytes
      - 解包 npz -> 提取 'arr'（或第一个数组）
      - 打印 shape 与部分内容到控制台
      - 保存为 .npy / .csv
      - 生成 heatmap png 并保存
      - 返回 JSON（包含保存路径与部分预览）
    """
    meta_json = request.form.get("meta")
    if not meta_json or "file" not in request.files:
        return jsonify({"ok": False, "msg": "missing meta or file"}), 400

    try:
        meta = json.loads(meta_json)
    except Exception:
        meta = {}

    file_storage = request.files["file"]
    file_bytes = file_storage.read()
    # 校验 sha256（如果 meta 中提供）
    provided_sha = meta.get("sha256")
    computed_sha = sha256_bytes(file_bytes)
    sha_ok = (provided_sha is None) or (provided_sha == computed_sha)

    part_id = meta.get("part_id", f"part_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    base_name = part_id

    # 先保存原始 npz 二进制，便于日后分析
    raw_path = os.path.join(SAVE_DIR, f"{base_name}.npz")
    with open(raw_path, "wb") as f:
        f.write(file_bytes)

    # 尝试从 npz 中加载矩阵
    arr = None
    try:
        bio = io.BytesIO(file_bytes)
        npz = np.load(bio, allow_pickle=False)
        # 尝试取名为 'arr' 的数组，否则拿第一个数组
        if 'arr' in npz.files:
            arr = npz['arr']
        else:
            # 取第一个文件里的数组
            first_key = npz.files[0] if len(npz.files) > 0 else None
            if first_key:
                arr = npz[first_key]
            else:
                arr = None
    except Exception as e:
        # 若不是 npz，尝试直接用 numpy.frombuffer 恢复（不常用）
        try:
            arr = np.frombuffer(file_bytes, dtype=np.float64)
        except Exception:
            arr = None
        load_error = str(e)
    else:
        load_error = None

    if arr is None:
        # 无法解析出的数组，返回错误
        return jsonify({
            "ok": False,
            "msg": "Cannot decode received file as numpy array",
            "raw_path": raw_path,
            "load_error": load_error
        }), 400

    # 确保为2D数组以便可视化，若为1D则reshape为列向量
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # 打印到控制台（只打印前几行以免刷屏）
    print(f"[EDGE] Received part_id={part_id}, shape={arr.shape}, sha_ok={sha_ok}")
    # 打印前5行与统计信息
    np.set_printoptions(precision=4, suppress=True)
    preview_rows = min(arr.shape[0], 12)
    print("[EDGE] preview rows:")
    print(arr[:preview_rows, :min(arr.shape[1], 8)])  # 显示前8列
    print("[EDGE] stats: min=%g, max=%g, mean=%g" % (arr.min(), arr.max(), arr.mean()))

    # 保存并可视化
    saved_info = _save_matrix_and_visual(arr, base_name)
    saved_info.update({
        "ok": True,
        "part_id": part_id,
        "raw_path": raw_path,
        "shape": arr.shape,
        "sha_ok": sha_ok
    })

    # 返回保存路径等信息给主节点
    return jsonify(saved_info)

@app.route("/visual/<part_name>", methods=["GET"])
def get_visual(part_name: str):
    """
    额外提供一个接口直接下载 heatmap PNG（如果存在）。
    URL 示例: GET /visual/edge_0_heatmap.png
    """
    png_path = os.path.join(SAVE_DIR, part_name)
    if not os.path.isfile(png_path):
        return jsonify({"ok": False, "msg": "not found"}), 404
    return send_file(png_path, mimetype="image/png")

if __name__ == "__main__":
    # 监听所有网卡（主节点可以访问），端口根据需要设定
    app.run(host="0.0.0.0", port=5000)