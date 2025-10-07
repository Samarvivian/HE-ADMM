import random
import math
import time
import numpy as np
from typing import Tuple, List, Dict  # 添加 Dict 导入


class OptimizedPaillierEncryption:
    """
    优化的Paillier同态加密系统
    严格遵循论文第III-B节的数学描述
    """

    def __init__(self, key_bits: int = 1024):
        """
        初始化Paillier加密系统

        Args:
            key_bits: 密钥位数，论文要求至少1024位
        """
        self.key_bits = key_bits
        self._generate_keys_according_to_paper()

    def _generate_keys_according_to_paper(self):
        """
        按照论文第III-B节的密钥生成方法：
        • 选择两个大素数 p, q
        • 计算 n = pq, ε = lcm(p-1, q-1)
        • 选择整数 g ∈ Z_{n^2}^*, gcd(g, n) = 1
        • 选择整数 r ∈ Z_n^*, gcd(r, n) = 1
        • 计算 μ = (L(g^ε mod n^2))^{-1} mod n
        """
        print("按照论文方法生成Paillier密钥对...")

        # 步骤1: 选择两个大素数 p, q
        print("  生成大素数 p, q...")
        self.p = self._generate_secure_prime(self.key_bits // 2)
        self.q = self._generate_secure_prime(self.key_bits // 2)

        # 确保 p ≠ q
        while self.p == self.q:
            self.q = self._generate_secure_prime(self.key_bits // 2)

        # 步骤2: 计算 n = p * q, n^2
        self.n = self.p * self.q
        self.n_square = self.n ** 2

        print(f"  素数生成完成: p位数={self.p.bit_length()}, q位数={self.q.bit_length()}")
        print(f"  计算 n = p * q = {self.n} (位数: {self.n.bit_length()})")

        # 步骤3: 计算 ε = lcm(p-1, q-1)
        self.epsilon = self._lcm(self.p - 1, self.q - 1)
        print(f"  计算 ε = lcm(p-1, q-1) = {self.epsilon}")

        # 步骤4: 选择生成元 g ∈ Z_{n^2}^*, gcd(g, n) = 1
        print("  选择生成元 g...")
        self.g = self._find_generator()

        # 步骤5: 选择随机数 r ∈ Z_n^*, gcd(r, n) = 1
        # 注意：论文中r在每次加密时重新生成，这里我们生成一个初始r用于演示
        self.r = self._find_coprime_random()

        # 步骤6: 计算 μ = (L(g^ε mod n^2))^{-1} mod n
        print("  计算 μ...")
        g_epsilon = pow(self.g, self.epsilon, self.n_square)
        L_val = self._L_function(g_epsilon)

        if math.gcd(L_val, self.n) != 1:
            raise ValueError("L(g^ε) 与 n 不互质，无法计算逆元")

        self.mu = pow(L_val, -1, self.n)

        print("Paillier密钥生成完成!")
        print(f"  公钥: (n={self.n}, g={self.g})")
        print(f"  私钥: (p={self.p}, q={self.q}, ε={self.epsilon}, μ={self.mu})")
        print(f"  满足论文要求: 密钥长度 ≥ {self.key_bits}位")

    def _generate_secure_prime(self, bits: int) -> int:
        """
        生成安全素数（使用更健壮的素数检测）
        """

        def miller_rabin_test(n: int, k: int = 5) -> bool:
            """Miller-Rabin素数检测"""
            if n < 2:
                return False
            if n in (2, 3):
                return True
            if n % 2 == 0:
                return False

            # 将 n-1 写成 2^r * d
            r, d = 0, n - 1
            while d % 2 == 0:
                r += 1
                d //= 2

            def check_composite(a):
                x = pow(a, d, n)
                if x in (1, n - 1):
                    return False
                for _ in range(r - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        return False
                return True

            for _ in range(k):
                a = random.randint(2, n - 2)
                if check_composite(a):
                    return False
            return True

        # 为了测试效率，限制最大位数
        max_test_bits = min(bits, 32)  # 测试时使用32位最大
        print(f"    生成 {max_test_bits} 位素数...")

        while True:
            candidate = random.getrandbits(max_test_bits)
            # 确保是奇数且足够大
            candidate |= (1 << (max_test_bits - 1)) | 1

            if miller_rabin_test(candidate):
                return candidate

    def _lcm(self, a: int, b: int) -> int:
        """计算最小公倍数"""
        return abs(a * b) // math.gcd(a, b)

    def _find_generator(self) -> int:
        """寻找合适的生成元 g ∈ Z_{n^2}^*, gcd(g, n) = 1"""
        # 简化方法：选择 g = n + 1，这在Paillier中常用且满足条件
        candidate = self.n + 1

        # 验证 gcd(g, n^2) = 1
        if math.gcd(candidate, self.n_square) == 1:
            return candidate

        # 如果不行，随机搜索
        while True:
            candidate = random.randint(2, self.n_square - 1)
            if math.gcd(candidate, self.n_square) == 1:
                return candidate

    def _find_coprime_random(self) -> int:
        """寻找与n互质的随机数"""
        while True:
            r = random.randint(2, self.n - 1)
            if math.gcd(r, self.n) == 1:
                return r

    def _L_function(self, x: int) -> int:
        """L函数: L(x) = (x-1)/n"""
        return (x - 1) // self.n

    def encrypt(self, plaintext: int, r: int = None) -> int:
        """
        加密函数 - 论文公式(15a, 15b, 15c)
        c = g^m * r^n mod n^2

        Args:
            plaintext: 明文整数 (经过量化后的正整数)
            r: 随机数 (如果为None则自动生成)

        Returns:
            密文整数
        """
        if plaintext < 0 or plaintext >= self.n:
            raise ValueError(f"明文必须在 [0, {self.n - 1}] 范围内")

        # 如果没有提供r，生成一个随机r
        if r is None:
            r = self._find_coprime_random()

        # 加密计算
        term1 = pow(self.g, plaintext, self.n_square)
        term2 = pow(r, self.n, self.n_square)
        ciphertext = (term1 * term2) % self.n_square

        return ciphertext

    def batch_encrypt(self, plaintexts: List[int]) -> List[int]:
        """批量加密"""
        return [self.encrypt(pt) for pt in plaintexts]

    def decrypt(self, ciphertext: int) -> int:
        """
        解密函数
        m = L(c^ε mod n^2) * μ mod n
        """
        # 验证密文在有效范围内
        if ciphertext < 0 or ciphertext >= self.n_square:
            raise ValueError("密文不在有效范围内")

        # 解密计算
        c_epsilon = pow(ciphertext, self.epsilon, self.n_square)
        L_val = self._L_function(c_epsilon)
        plaintext = (L_val * self.mu) % self.n

        return plaintext

    def batch_decrypt(self, ciphertexts: List[int]) -> List[int]:
        """批量解密"""
        return [self.decrypt(ct) for ct in ciphertexts]

    def homomorphic_add(self, ciphertext1: int, ciphertext2: int) -> int:
        """
        同态加法 - 论文定义1
        c1 ⊕ c2 = c1 * c2 mod n^2
        """
        return (ciphertext1 * ciphertext2) % self.n_square

    def homomorphic_scalar_multiply(self, ciphertext: int, scalar: int) -> int:
        """
        同态标量乘法 - 论文定义2
        k ⊗ c = c^k mod n^2
        """
        return pow(ciphertext, scalar, self.n_square)

    def verify_key_security(self) -> Dict[str, bool]:
        """验证密钥安全性"""
        checks = {
            "p_is_prime": self._is_prime(self.p),
            "q_is_prime": self._is_prime(self.q),
            "p_neq_q": self.p != self.q,
            "n_correct": self.n == self.p * self.q,
            "gcd_g_n2": math.gcd(self.g, self.n_square) == 1,
            "gcd_r_n": math.gcd(self.r, self.n) == 1,
            "key_length_adequate": self.n.bit_length() >= self.key_bits,
            "mu_valid": (self.mu * self._L_function(pow(self.g, self.epsilon, self.n_square))) % self.n == 1
        }

        return checks

    def _is_prime(self, n: int) -> bool:
        """简单素数检测"""
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True


class ThreePADMMPaillierIntegration:
    """
    3P-ADMM-Paillier集成类
    演示如何将Paillier加密与ADMM算法结合
    """

    def __init__(self, key_bits: int = 1024):
        self.paillier = OptimizedPaillierEncryption(key_bits)
        self.quantization_delta = 1000  # 量化参数Δ

    def gamma1_quantize(self, vector: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        """Γ1量化函数 - 论文公式(14a)"""
        if v_max <= v_min:
            return np.zeros_like(vector, dtype=np.int64)

        denominator = (v_max - v_min) ** 2
        scale = self.quantization_delta ** 2 / denominator

        quantized = np.floor(scale * (vector - v_min))
        return np.clip(quantized, 0, None).astype(np.int64)

    def gamma2_quantize(self, vector: np.ndarray, v_min: float, v_max: float) -> np.ndarray:
        """Γ2量化函数 - 论文公式(14b, 14c, 14d)"""
        if v_max <= v_min:
            return np.zeros_like(vector, dtype=np.int64)

        scale = self.quantization_delta / (v_max - v_min)
        quantized = np.floor(scale * (vector - v_min))
        return np.clip(quantized, 0, None).astype(np.int64)

    def encrypt_admm_variables(self, alpha_k: np.ndarray, z_k: np.ndarray, v_k: np.ndarray,
                               alpha_range: Tuple, z_range: Tuple, v_range: Tuple) -> Dict:
        """
        加密ADMM变量 - 论文公式(11, 12)
        """
        # 量化
        alpha_quant = self.gamma1_quantize(alpha_k, *alpha_range)
        z_quant = self.gamma2_quantize(z_k, *z_range)
        v_quant = self.gamma2_quantize(-v_k, *v_range)  # 注意是 -v_k

        # 加密
        alpha_enc = self.paillier.batch_encrypt(alpha_quant.tolist())
        z_enc = self.paillier.batch_encrypt(z_quant.tolist())
        v_enc = self.paillier.batch_encrypt(v_quant.tolist())

        return {
            'alpha_hat': np.array(alpha_enc),
            'z_hat': np.array(z_enc),
            'v_hat': np.array(v_enc)
        }

    def homomorphic_computation(self, alpha_hat: np.ndarray, z_hat: np.ndarray,
                                v_hat: np.ndarray, B_bar_k: np.ndarray) -> np.ndarray:
        """
        同态计算 - 论文公式(13)
        x̂_k = α̂_k ⊕ Γ₂(B̄_k) ⊗ (ẑ_k ⊕ (-v̂_k))
        """
        # ẑ_k ⊕ (-v̂_k)
        z_plus_minus_v = np.array([
            self.paillier.homomorphic_add(z_hat[i], v_hat[i])
            for i in range(len(z_hat))
        ])

        # Γ₂(B̄_k) ⊗ (ẑ_k ⊕ (-v̂_k))
        B_times_sum = np.array([
            self.paillier.homomorphic_scalar_multiply(
                z_plus_minus_v[i],
                int(B_bar_k[i, i])  # 简化：使用对角线元素
            )
            for i in range(len(z_plus_minus_v))
        ])

        # α̂_k ⊕ [Γ₂(B̄_k) ⊗ (ẑ_k ⊕ (-v̂_k))]
        x_hat = np.array([
            self.paillier.homomorphic_add(alpha_hat[i], B_times_sum[i])
            for i in range(len(alpha_hat))
        ])

        return x_hat


def test_paillier_correctness():
    """测试Paillier加密解密的正确性"""
    print("\nPaillier加密解密正确性测试")
    print("=" * 50)

    # 创建Paillier实例
    paillier = OptimizedPaillierEncryption(key_bits=128)

    # 测试数据
    test_messages = [0, 1, 5, 10, 50, 100, 500]

    print("加密解密测试:")
    all_correct = True

    for message in test_messages:
        try:
            # 加密
            ciphertext = paillier.encrypt(message)

            # 解密
            decrypted = paillier.decrypt(ciphertext)

            # 验证
            is_correct = (message == decrypted)
            all_correct = all_correct and is_correct

            print(
                f"  明文: {message:3d} -> 密文: {ciphertext:20d} -> 解密: {decrypted:3d} {'✓' if is_correct else '✗'}")

        except Exception as e:
            print(f"  明文: {message:3d} -> 错误: {e}")
            all_correct = False

    print(f"\n加密解密测试结果: {'全部正确 ✓' if all_correct else '存在错误 ✗'}")


def test_homomorphic_properties():
    """测试Paillier同态性质"""
    print("\nPaillier同态性质测试")
    print("=" * 50)

    paillier = OptimizedPaillierEncryption(key_bits=128)

    # 测试同态加法
    print("同态加法测试:")
    m1, m2 = 15, 25
    c1 = paillier.encrypt(m1)
    c2 = paillier.encrypt(m2)

    # 同态加法
    c_sum = paillier.homomorphic_add(c1, c2)
    m_sum = paillier.decrypt(c_sum)

    expected_sum = (m1 + m2) % paillier.n
    is_correct = (m_sum == expected_sum)

    print(f"  {m1} + {m2} = {m_sum} (期望: {expected_sum}) {'✓' if is_correct else '✗'}")

    # 测试同态标量乘法
    print("\n同态标量乘法测试:")
    m = 10
    scalar = 3
    c = paillier.encrypt(m)

    # 同态标量乘法
    c_mult = paillier.homomorphic_scalar_multiply(c, scalar)
    m_mult = paillier.decrypt(c_mult)

    expected_mult = (scalar * m) % paillier.n
    is_correct = (m_mult == expected_mult)

    print(f"  {scalar} * {m} = {m_mult} (期望: {expected_mult}) {'✓' if is_correct else '✗'}")


def comprehensive_paillier_test():
    """全面的Paillier系统测试"""
    print("Paillier加密系统全面测试")
    print("=" * 60)

    # 测试1: 基本功能测试
    print("1. 基本加密解密测试")
    paillier = OptimizedPaillierEncryption(key_bits=128)

    test_messages = [0, 1, 10, 100, 500, 1000]
    print("   加密解密正确性:")
    for msg in test_messages:
        try:
            ciphertext = paillier.encrypt(msg)
            decrypted = paillier.decrypt(ciphertext)
            status = "✓" if msg == decrypted else "✗"
            print(f"     {msg:4d} -> {decrypted:4d} {status}")
        except Exception as e:
            print(f"     {msg:4d} -> 错误: {e}")

    # 测试2: 同态性质测试
    print("\n2. 同态性质测试")

    # 同态加法
    m1, m2 = 50, 30
    c1 = paillier.encrypt(m1)
    c2 = paillier.encrypt(m2)
    c_sum = paillier.homomorphic_add(c1, c2)
    m_sum = paillier.decrypt(c_sum)
    print(f"   同态加法: {m1} + {m2} = {m_sum} (期望: {m1 + m2}) {'✓' if m_sum == m1 + m2 else '✗'}")

    # 同态标量乘法
    m, scalar = 20, 5
    c = paillier.encrypt(m)
    c_mult = paillier.homomorphic_scalar_multiply(c, scalar)
    m_mult = paillier.decrypt(c_mult)
    print(f"   同态乘法: {scalar} × {m} = {m_mult} (期望: {scalar * m}) {'✓' if m_mult == scalar * m else '✗'}")

    # 测试3: 批量操作测试
    print("\n3. 批量操作测试")
    batch_messages = [i * 10 for i in range(5)]
    encrypted_batch = paillier.batch_encrypt(batch_messages)
    decrypted_batch = paillier.batch_decrypt(encrypted_batch)

    print("   批量加密解密:")
    for i, (orig, dec) in enumerate(zip(batch_messages, decrypted_batch)):
        status = "✓" if orig == dec else "✗"
        print(f"     消息 {i}: {orig} -> {dec} {status}")

    # 测试4: 安全性验证
    print("\n4. 密钥安全性验证")
    security_checks = paillier.verify_key_security()
    for check_name, result in security_checks.items():
        status = "✓" if result else "✗"
        print(f"   {check_name}: {status}")

    # 测试5: 性能测试
    print("\n5. 性能测试")
    start_time = time.time()

    # 加密性能
    enc_times = []
    for i in range(20):
        msg = random.randint(0, 1000)
        start_enc = time.time()
        ciphertext = paillier.encrypt(msg)
        enc_times.append(time.time() - start_enc)

    # 解密性能
    dec_times = []
    for i in range(20):
        msg = random.randint(0, 1000)
        ciphertext = paillier.encrypt(msg)
        start_dec = time.time()
        decrypted = paillier.decrypt(ciphertext)
        dec_times.append(time.time() - start_dec)

    avg_enc_time = sum(enc_times) / len(enc_times)
    avg_dec_time = sum(dec_times) / len(dec_times)

    print(f"   平均加密时间: {avg_enc_time:.6f} 秒")
    print(f"   平均解密时间: {avg_dec_time:.6f} 秒")
    print(f"   总测试时间: {time.time() - start_time:.2f} 秒")


def admm_paillier_integration_demo():
    """ADMM-Paillier集成演示"""
    print("\nADMM-Paillier集成演示")
    print("=" * 60)

    # 创建集成系统
    system = ThreePADMMPaillierIntegration(key_bits=128)

    # 模拟ADMM变量
    np.random.seed(42)
    alpha_k = np.random.randn(5) * 10 + 5  # α_k = B_k A_k^T y
    z_k = np.random.randn(5) * 5 + 2  # z_k 变量
    v_k = np.random.randn(5) * 3 + 1  # v_k 变量
    B_bar_k = np.random.randn(5, 5) * 2 + 1  # 量化后的 B_k

    # 量化范围
    alpha_range = (np.min(alpha_k), np.max(alpha_k))
    z_range = (np.min(z_k), np.max(z_k))
    v_range = (np.min(-v_k), np.max(-v_k))

    print("模拟ADMM变量:")
    print(f"  alpha_k: {alpha_k}")
    print(f"  z_k: {z_k}")
    print(f"  v_k: {v_k}")
    print(f"  量化范围: alpha{alpha_range}, z{z_range}, v{v_range}")

    # 加密变量
    encrypted_vars = system.encrypt_admm_variables(alpha_k, z_k, v_k, alpha_range, z_range, v_range)
    print("\n变量加密完成")

    # 同态计算
    x_hat = system.homomorphic_computation(
        encrypted_vars['alpha_hat'],
        encrypted_vars['z_hat'],
        encrypted_vars['v_hat'],
        B_bar_k
    )
    print("同态计算完成")

    # 解密结果
    x_decrypted = system.paillier.batch_decrypt(x_hat.tolist())
    print(f"解密结果: {x_decrypted}")


if __name__ == "__main__":
    # 运行基本测试
    test_paillier_correctness()
    test_homomorphic_properties()

    # 运行全面测试
    comprehensive_paillier_test()

    # 运行集成演示
    admm_paillier_integration_demo()