# test_protocol.py - 协议一致性测试
import socket
import time


class ProtocolTester:
    def __init__(self, host='raspberrypi.local', port=8888):
        self.host = host
        self.port = port

    def send_command(self, command):
        """发送命令并返回响应"""
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(10)
            client.connect((self.host, self.port))

            client.send(command.encode('utf-8'))
            response = client.recv(1024).decode('utf-8')
            client.close()

            return response
        except Exception as e:
            return f"ERROR:Connection failed: {str(e)}"

    def run_tests(self):
        """运行完整的协议测试套件"""
        test_cases = [
            # (指令, 描述)
            ("PING", "心跳测试"),
            ("QUANTIZE:3.14,-5.0,5.0,1000000", "浮点数量化测试"),
            ("INVERSE_QUANTIZE:750000,-5.0,5.0,1000000", "反量化测试"),
            ("ENCRYPT:2500000", "加密测试"),
            ("DECRYPT:2500000123456", "解密测试"),
            ("HOMO_ADD:891234567890,912345678901", "同态加法测试"),
            ("HOMO_MUL_CONST:891234567890,3", "同态乘常数测试"),
            ("COMPUTE_LOCAL_UPDATE:1000,2000,3000,1.5", "ADMM本地更新测试"),
        ]

        print("🚀 开始HE-ADMM通信协议测试")
        print("=" * 50)

        for i, (command, description) in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {description}")
            print(f"   发送: {command}")

            response = self.send_command(command)
            print(f"   接收: {response}")

            time.sleep(0.5)  # 避免请求过于频繁

        print("\n" + "=" * 50)
        print("✅ 协议测试完成！")


if __name__ == "__main__":
    # 使用树莓派的hostname或IP地址
    tester = ProtocolTester(host='raspberrypi.local', port=8888)
    tester.run_tests()