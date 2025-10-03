# edge_server.py - 基于HE-ADMM通信协议v1.0
import socket
import threading
import logging
from datetime import datetime

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [边缘节点] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edge_node.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class EdgeNodeServer:
    def __init__(self, host='0.0.0.0', port=8888):
        self.host = host
        self.port = port
        self.running = False

    def handle_quantize(self, params):
        """处理量化指令: QUANTIZE:浮点数,最小值,最大值,Delta"""
        if len(params) != 4:
            return "ERROR:Invalid parameters, expected: float_val,min_val,max_val,delta"

        try:
            float_val, min_val, max_val, delta = map(float, params)
            # TODO: 调用A同学的量化函数
            # quantized = A_quantize(float_val, min_val, max_val, delta)
            # 模拟实现
            normalized = (float_val - min_val) / (max_val - min_val)
            quantized = int(normalized * delta)
            logging.info(f"量化: {float_val} -> {quantized}")
            return f"SUCCESS:{quantized}"
        except Exception as e:
            return f"ERROR:Quantization failed: {str(e)}"

    def handle_inverse_quantize(self, params):
        """处理反量化指令: INVERSE_QUANTIZE:量化值,最小值,最大值,Delta"""
        if len(params) != 4:
            return "ERROR:Invalid parameters, expected: quantized_val,min_val,max_val,delta"

        try:
            quantized_val, min_val, max_val, delta = map(float, params)
            # TODO: 调用A同学的反量化函数
            # original = A_inverse_quantize(quantized_val, min_val, max_val, delta)
            # 模拟实现
            normalized = quantized_val / delta
            original = normalized * (max_val - min_val) + min_val
            logging.info(f"反量化: {quantized_val} -> {original:.6f}")
            return f"SUCCESS:{original:.6f}"
        except Exception as e:
            return f"ERROR:Inverse quantization failed: {str(e)}"

    def handle_encrypt(self, params):
        """处理加密指令: ENCRYPT:明文整数"""
        if len(params) != 1:
            return "ERROR:Invalid parameters, expected: plaintext_int"

        try:
            plaintext = int(params[0])
            # TODO: 调用B同学的加密函数
            # ciphertext = B_encrypt(plaintext)
            # 模拟实现 - 返回一个"看起来像"密文的大数字
            ciphertext = plaintext * 1000000 + 123456  # 模拟加密
            logging.info(f"加密: {plaintext} -> {ciphertext}")
            return f"SUCCESS:{ciphertext}"
        except Exception as e:
            return f"ERROR:Encryption failed: {str(e)}"

    def handle_decrypt(self, params):
        """处理解密指令: DECRYPT:密文"""
        if len(params) != 1:
            return "ERROR:Invalid parameters, expected: ciphertext"

        try:
            ciphertext = int(params[0])
            # TODO: 调用B同学的解密函数
            # plaintext = B_decrypt(ciphertext)
            # 模拟实现 - 从"密文"中恢复明文
            plaintext = (ciphertext - 123456) // 1000000  # 模拟解密
            logging.info(f"解密: {ciphertext} -> {plaintext}")
            return f"SUCCESS:{plaintext}"
        except Exception as e:
            return f"ERROR:Decryption failed: {str(e)}"

    def handle_homo_add(self, params):
        """处理同态加法: HOMO_ADD:密文1,密文2"""
        if len(params) != 2:
            return "ERROR:Invalid parameters, expected: ciphertext1,ciphertext2"

        try:
            cipher1, cipher2 = map(int, params)
            # TODO: 调用B同学的同态加法函数
            # result = B_homomorphic_add(cipher1, cipher2)
            # 模拟实现 - Paillier同态加法是乘法
            result = cipher1 * cipher2
            logging.info(f"同态加法: {cipher1} * {cipher2} = {result}")
            return f"SUCCESS:{result}"
        except Exception as e:
            return f"ERROR:Homomorphic addition failed: {str(e)}"

    def handle_homo_mul_const(self, params):
        """处理同态乘常数: HOMO_MUL_CONST:密文,常数"""
        if len(params) != 2:
            return "ERROR:Invalid parameters, expected: ciphertext,constant"

        try:
            ciphertext, constant = map(int, params)
            # TODO: 调用B同学的同态乘常数函数
            # result = B_homomorphic_multiply_constant(ciphertext, constant)
            # 模拟实现 - Paillier同态乘常数是幂运算
            result = ciphertext ** constant
            logging.info(f"同态乘常数: {ciphertext}^{constant} = {result}")
            return f"SUCCESS:{result}"
        except Exception as e:
            return f"ERROR:Homomorphic multiplication failed: {str(e)}"

    def handle_compute_local_update(self, params):
        """处理ADMM本地更新: COMPUTE_LOCAL_UPDATE:加密观测向量,加密z向量,加密v向量,rho参数"""
        if len(params) != 4:
            return "ERROR:Invalid parameters, expected: enc_alpha,enc_z,enc_v,rho"

        try:
            enc_alpha, enc_z, enc_v, rho = params
            enc_alpha, enc_z, enc_v = map(int, [enc_alpha, enc_z, enc_v])
            rho = float(rho)

            # TODO: 调用A和B同学的函数完成(13)式计算
            # 模拟实现 - 这里应该执行: enc_alpha + enc_Bk * (enc_z + (-enc_v))
            # 简化模拟：直接返回一个"更新后"的密文
            result = enc_alpha + enc_z - enc_v  # 简化的模拟计算
            logging.info(f"本地ADMM更新: alpha={enc_alpha}, z={enc_z}, v={enc_v}, rho={rho} -> {result}")
            return f"SUCCESS:{result}"
        except Exception as e:
            return f"ERROR:Local ADMM computation failed: {str(e)}"

    def process_command(self, request):
        """处理客户端请求"""
        try:
            request = request.strip()
            logging.info(f"收到请求: {request}")

            # 解析指令和参数
            if ':' in request:
                command, param_str = request.split(':', 1)
                params = [p.strip() for p in param_str.split(',')]
            else:
                command = request
                params = []

            # 根据协议路由到对应的处理函数
            command_handlers = {
                'PING': lambda p: "SUCCESS:PONG",
                'QUANTIZE': self.handle_quantize,
                'INVERSE_QUANTIZE': self.handle_inverse_quantize,
                'ENCRYPT': self.handle_encrypt,
                'DECRYPT': self.handle_decrypt,
                'HOMO_ADD': self.handle_homo_add,
                'HOMO_MUL_CONST': self.handle_homo_mul_const,
                'COMPUTE_LOCAL_UPDATE': self.handle_compute_local_update,
            }

            if command in command_handlers:
                return command_handlers[command](params)
            else:
                return "ERROR:Unknown command"

        except Exception as e:
            logging.error(f"处理命令时出错: {e}")
            return f"ERROR:Server error: {str(e)}"

    def handle_client_connection(self, client_socket):
        """处理客户端连接"""
        try:
            request = client_socket.recv(1024).decode('utf-8')
            response = self.process_command(request)
            client_socket.send(response.encode('utf-8'))
            logging.info(f"发送响应: {response}")
        except Exception as e:
            error_msg = f"ERROR:Connection error: {str(e)}"
            client_socket.send(error_msg.encode('utf-8'))
            logging.error(error_msg)
        finally:
            client_socket.close()

    def start(self):
        """启动边缘节点服务器"""
        self.running = True
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)

        logging.info(f"边缘节点服务启动在 {self.host}:{self.port}")
        logging.info(
            "支持指令: PING, QUANTIZE, INVERSE_QUANTIZE, ENCRYPT, DECRYPT, HOMO_ADD, HOMO_MUL_CONST, COMPUTE_LOCAL_UPDATE")

        try:
            while self.running:
                client_sock, address = server.accept()
                logging.info(f"接受来自 {address} 的连接")

                client_handler = threading.Thread(
                    target=self.handle_client_connection,
                    args=(client_sock,)
                )
                client_handler.start()

        except KeyboardInterrupt:
            logging.info("收到中断信号，关闭服务器")
        finally:
            server.close()
            logging.info("边缘节点服务已关闭")


if __name__ == "__main__":
    server = EdgeNodeServer()
    server.start()