# edge_server.py - 【增强版：支持同态加法】
import socket
import threading


def handle_client_connection(client_socket):
    """
    处理主节点发来的请求
    """
    try:
        # 接收数据 (假设数据是字符串格式的指令)
        request = client_socket.recv(1024).decode('utf-8').strip()
        print(f"[边缘节点] 收到指令: {request}")

        # 初始化响应
        response = "ERROR: Unknown command"

        # 1. 心跳检测指令
        if request == "PING":
            response = "PONG"

        # 2. 同态加法指令 (格式: HOMO_ADD:cipher1,cipher2)
        elif request.startswith("HOMO_ADD:"):
            # 解析指令，获取两个密文
            parts = request.split(':')[1].split(',')
            if len(parts) == 2:
                # ！！！核心同态操作！！！
                # 在Paillier加密中，同态加法就是密文相乘
                # 注意：这里我们假设传过来的是整数，模拟密文
                # 后期A同学会把它换成真正的Paillier大整数
                cipher1 = int(parts[0])
                cipher2 = int(parts[1])
                result_cipher = cipher1 * cipher2  # 同态加！！！

                response = f"HOMO_ADD_RESULT:{result_cipher}"
                print(f"[边缘节点] 同态加法: {cipher1} * {cipher2} = {result_cipher}")
            else:
                response = "ERROR: Invalid HOMO_ADD format"

        # 3. 回声测试指令 (用于调试数据传输)
        elif request.startswith("ECHO:"):
            data = request.split(':', 1)[1]
            response = f"ECHO_RESPONSE:{data}"

        # 发送回复
        client_socket.send(response.encode('utf-8'))
        print(f"[边缘节点] 已回复: {response}")

    except Exception as e:
        print(f"[边缘节点] 处理请求时出错: {e}")
    finally:
        client_socket.close()


def start_server(host='0.0.0.0', port=8888):
    """
    启动边缘节点服务
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)  # 最大连接数
    print(f"[边缘节点] 服务已启动，在 {host}:{port} 监听...")
    print(f"[边缘节点] 可用指令: PING, ECHO:<message>, HOMO_ADD:<num1>,<num2>")

    try:
        while True:
            client_sock, address = server.accept()
            print(f"[边缘节点] 接受来自 {address} 的连接")
            # 为每个连接创建新线程处理
            client_handler = threading.Thread(
                target=handle_client_connection,
                args=(client_sock,)
            )
            client_handler.start()
    except KeyboardInterrupt:
        print("\n[边缘节点] 服务器被手动关闭")
    finally:
        server.close()


if __name__ == "__main__":
    # 启动服务器
    start_server()