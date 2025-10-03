# test_master.py - 主节点测试客户端
import socket


def test_edge_server(host, port):
    """
    测试边缘节点服务
    """
    try:
        # 创建Socket连接
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((host, port))
        print(f"[主节点] 已连接到边缘节点 {host}:{port}")

        # 测试1: 心跳检测
        print("\n1. 测试心跳...")
        client.send("PING".encode())
        response = client.recv(1024).decode()
        print(f"   发送: PING -> 接收: {response}")

        # 测试2: 回声测试
        print("\n2. 测试回声...")
        test_message = "Hello_Raspberry_Pi"
        client.send(f"ECHO:{test_message}".encode())
        response = client.recv(1024).decode()
        print(f"   发送: ECHO:{test_message} -> 接收: {response}")

        # 测试3: 同态加法测试 (模拟!)
        print("\n3. 测试同态加法...")
        # 假设我们有两个"密文" 123 和 456
        # 同态加法结果应该是 123 * 456 = 56088
        client.send("HOMO_ADD:123,456".encode())
        response = client.recv(1024).decode()
        print(f"   发送: HOMO_ADD:123,456 -> 接收: {response}")

        client.close()
        print(f"\n[主节点] 所有测试完成！")

    except Exception as e:
        print(f"[主节点] 连接测试失败: {e}")


if __name__ == "__main__":
    # ！！！重要：把这里的IP换成你树莓派的实际IP！！！
    EDGE_NODE_IP = "192.168.214.223"  # 替换成你的树莓派IP!
    EDGE_NODE_PORT = 8888

    test_edge_server(EDGE_NODE_IP, EDGE_NODE_PORT)