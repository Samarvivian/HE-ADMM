#!/usr/bin/env python3
"""
修复Edge节点和主节点通信问题的脚本
"""

import os
import subprocess
import sys

def check_network_connectivity():
    """检查网络连通性"""
    print("=== 检查网络连通性 ===")
    
    # 检查主节点连通性
    master_ip = "192.168.201.154"
    edge_ip = "192.168.201.223"
    
    print(f"检查主节点 {master_ip} 连通性...")
    result = subprocess.run(['ping', '-c', '1', master_ip], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ 主节点 {master_ip} 可达")
    else:
        print(f"✗ 主节点 {master_ip} 不可达")
        print("请检查:")
        print("1. 主节点是否开机")
        print("2. 网络连接是否正常")
        print("3. IP地址是否正确")
    
    print(f"\n检查Edge节点 {edge_ip} 连通性...")
    result = subprocess.run(['ping', '-c', '1', edge_ip], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✓ Edge节点 {edge_ip} 可达")
    else:
        print(f"✗ Edge节点 {edge_ip} 不可达")
        print("请检查:")
        print("1. Edge节点是否开机")
        print("2. 网络连接是否正常")
        print("3. IP地址是否正确")

def check_ports():
    """检查端口是否开放"""
    print("\n=== 检查端口开放情况 ===")
    
    import socket
    
    def check_port(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    # 检查主节点端口
    master_port = 8000
    if check_port("192.168.201.154", master_port):
        print(f"✓ 主节点端口 {master_port} 开放")
    else:
        print(f"✗ 主节点端口 {master_port} 未开放")
        print("请检查:")
        print("1. 主节点服务是否启动")
        print("2. 防火墙是否阻止了端口")
    
    # 检查Edge节点端口
    edge_port = 5000
    if check_port("192.168.201.223", edge_port):
        print(f"✓ Edge节点端口 {edge_port} 开放")
    else:
        print(f"✗ Edge节点端口 {edge_port} 未开放")
        print("请检查:")
        print("1. Edge节点服务是否启动")
        print("2. 防火墙是否阻止了端口")

def create_test_scripts():
    """创建测试脚本"""
    print("\n=== 创建测试脚本 ===")
    
    # 创建主节点测试脚本
    master_test = '''#!/usr/bin/env python3
import os
import io
import json
import hashlib
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
SAVE_DIR = "/home/pi/received_parts"
os.makedirs(SAVE_DIR, exist_ok=True)

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

@app.route("/receive_result", methods=["POST"])
def receive_result():
    print("收到请求!")
    meta_json = request.form.get("meta")
    if not meta_json or "file" not in request.files:
        print("缺少meta或file")
        return jsonify({"ok": False, "msg": "missing meta or file"}), 400

    meta = json.loads(meta_json)
    file_data = request.files["file"].read()
    part_id = meta.get("part_id", "unknown_part")
    save_path = os.path.join(SAVE_DIR, f"{part_id}.npz")
    with open(save_path, "wb") as f:
        f.write(file_data)

    ok = (sha256_bytes(file_data) == meta.get("sha256"))
    print(f"[MASTER] 已接收来自 {part_id} 的结果文件 -> {save_path}, 校验: {ok}")
    return jsonify({"ok": True, "msg": "result received", "path": save_path})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "主节点运行正常"})

if __name__ == "__main__":
    print("启动主节点测试服务...")
    app.run(host="0.0.0.0", port=8000, debug=True)
'''
    
    with open("master_test.py", "w", encoding="utf-8") as f:
        f.write(master_test)
    print("✓ 创建了 master_test.py")
    
    # 创建Edge节点测试脚本
    edge_test = '''#!/usr/bin/env python3
import os
import io
import json
import hashlib
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
SAVE_DIR = os.path.expanduser("~/received_parts")
os.makedirs(SAVE_DIR, exist_ok=True)

MASTER_URL = "http://192.168.201.154:8000/receive_result"

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _serialize_np_array(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.savez_compressed(bio, arr=arr)
    return bio.getvalue()

def send_result_to_master(result_array: np.ndarray, part_id: str):
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
    print(f"[EDGE] 主节点响应: {resp.status_code}, 内容: {resp.text}")
    return resp

@app.route("/upload", methods=["POST"])
def upload_matrix():
    print("收到上传请求!")
    meta_json = request.form.get("meta")
    if not meta_json or "file" not in request.files:
        print("缺少meta或file")
        return jsonify({"ok": False, "msg": "missing meta or file"}), 400

    try:
        meta = json.loads(meta_json)
    except Exception as e:
        print(f"解析meta失败: {e}")
        return jsonify({"ok": False, "msg": "invalid meta json"}), 400

    file_obj = request.files["file"]
    part_id = str(meta.get("part_id", "unknown_part"))
    
    # 保存文件
    save_path = os.path.join(SAVE_DIR, f"{part_id}.npz")
    file_bytes = file_obj.read()
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    print(f"[EDGE] 已接收文件并保存: {save_path} ({len(file_bytes)} bytes)")

    # 加载矩阵并计算简单结果
    try:
        with np.load(io.BytesIO(file_bytes)) as data:
            if "arr" in data:
                Ak = data["arr"]
            else:
                Ak = data[list(data.keys())[0]]
        
        # 简单的计算：Bk = Ak^T * Ak
        Bk = Ak.T @ Ak
        print(f"[EDGE] 计算完成，结果形状: {Bk.shape}")
        
    except Exception as e:
        print(f"[EDGE][ERROR] 计算失败: {e}")
        return jsonify({"ok": False, "msg": f"computation failed: {e}"}), 500

    try:
        send_result_to_master(Bk, part_id)
    except Exception as e:
        print(f"[EDGE][ERROR] 回传结果失败: {e}")
        return jsonify({"ok": False, "msg": f"send result failed: {e}"}), 502

    return jsonify({"ok": True, "msg": "Bk computed and sent"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Edge节点运行正常"})

if __name__ == "__main__":
    host = os.environ.get("EDGE_HOST", "0.0.0.0")
    port = int(os.environ.get("EDGE_PORT", "5000"))
    print(f"[EDGE] 启动服务: http://{host}:{port}  回传地址: {MASTER_URL}")
    app.run(host=host, port=port, debug=True)
'''
    
    with open("edge_test.py", "w", encoding="utf-8") as f:
        f.write(edge_test)
    print("✓ 创建了 edge_test.py")

def main():
    print("Edge节点通信问题诊断和修复工具")
    print("=" * 50)
    
    check_network_connectivity()
    check_ports()
    create_test_scripts()
    
    print("\n=== 修复建议 ===")
    print("1. 首先运行网络连通性检查")
    print("2. 如果网络不通，检查IP地址和网络配置")
    print("3. 如果网络通但端口不通，检查服务状态和防火墙")
    print("4. 使用创建的测试脚本进行简单测试:")
    print("   - 在主节点运行: python master_test.py")
    print("   - 在Edge节点运行: python edge_test.py")
    print("5. 测试成功后，再使用原始代码")

if __name__ == "__main__":
    main()
