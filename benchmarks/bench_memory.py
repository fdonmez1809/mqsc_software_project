import time
import gc
import tracemalloc
import numpy as np

# ----------------------------
# helper: timing
# ----------------------------
def bench_time(fn, iters=2000, warmup=200):
    gc.disable()
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    gc.enable()
    total = t1 - t0
    return total, total / iters

# ----------------------------
# helper: memory
# ----------------------------
def bench_memory(fn, iters=200):
    gc.disable()
    tracemalloc.start()
    for _ in range(iters):
        fn()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.enable()
    return current, peak

# ----------------------------
# wrappers around your code
# ----------------------------

# 1) Schoolbook RingLWE class: keep your exact code style, only wrap here
class RingLWE_schoolbook:
    def __init__(self, n, q, sigma):
        self.n = n
        self.q = q
        self.sigma = sigma

    def add_polynomials(self, p1, p2):
        return (p1 + p2) % self.q

    def subtract_polynomials(self, p1, p2):
        return (p1 - p2) % self.q

    def multiply_polynomials(self, p1, p2):
        initial_product = np.convolve(p1, p2).astype(int)

        while len(initial_product) > self.n:
            initial_degree = len(initial_product) - 1
            initial_coeff = initial_product[initial_degree]
            degree_mod_n = (initial_degree % n)
            n_amount = initial_degree // n
            final_coeff = pow(-1, n_amount)*initial_coeff
            initial_product = initial_product[:-1]
            initial_product[degree_mod_n] += final_coeff

        return initial_product % self.q

    def generate_error(self):
        error = np.random.normal(0, self.sigma, self.n)
        return np.round(error).astype(int) % self.q

    def generate_uniform(self):
        return np.random.randint(0, self.q, self.n)

    def generate_shared_secret(self):
        return self.generate_error()

    def encrypt(self, s, message_bits):
        a = self.generate_uniform()
        e = self.generate_error()
        message_poly = np.array(message_bits) * (self.q // 2)

        product = self.multiply_polynomials(a, s)
        b = self.add_polynomials(product, e)
        b = self.add_polynomials(b, message_poly)

        return a, b

    def decrypt(self, s, ciphertext):
        a, b = ciphertext
        product_as = self.multiply_polynomials(a, s)
        message_with_noise = self.subtract_polynomials(b, product_as)
        return self.remove_message_noise(message_with_noise)

    def remove_message_noise(self, message_with_noise):
        decrypted_bits = []
        for coefficient in message_with_noise:
            if (self.q // 4) < coefficient < (3 * self.q // 4):
                decrypted_bits.append(1)
            else:
                decrypted_bits.append(0)
        return decrypted_bits


# 2) NTT RingLWE class: uses your NTT_multiplication with fixed (n,q)
# IMPORTANT: assumes NTT_multiplication exists in the namespace
class RingLWE_ntt:
    def __init__(self, n, q, sigma):
        self.n = n
        self.q = q
        self.sigma = sigma

    def add_polynomials(self, p1, p2):
        return (p1 + p2) % self.q

    def subtract_polynomials(self, p1, p2):
        return (p1 - p2) % self.q

    def multiply_polynomials(self, p1, p2):
        result = NTT_multiplication(p1.tolist(), p2.tolist(), self.n, self.q)
        return np.array(result) % self.q

    def generate_error(self):
        error = np.random.normal(0, self.sigma, self.n)
        return np.round(error).astype(int) % self.q

    def generate_uniform(self):
        return np.random.randint(0, self.q, self.n)

    def generate_shared_secret(self):
        return self.generate_error()

    def encrypt(self, s, message_bits):
        a = self.generate_uniform()
        e = self.generate_error()
        message_poly = np.array(message_bits) * (self.q // 2)

        product = self.multiply_polynomials(a, s)
        b = self.add_polynomials(product, e)
        b = self.add_polynomials(b, message_poly)

        return a, b

    def decrypt(self, s, ciphertext):
        a, b = ciphertext
        product_as = self.multiply_polynomials(a, s)
        message_with_noise = self.subtract_polynomials(b, product_as)
        return self.remove_message_noise(message_with_noise)

    def remove_message_noise(self, message_with_noise):
        decrypted_bits = []
        for coefficient in message_with_noise:
            if (self.q // 4) < coefficient < (3 * self.q // 4):
                decrypted_bits.append(1)
            else:
                decrypted_bits.append(0)
        return decrypted_bits


# ----------------------------
# run benchmark
# ----------------------------
def run_all(n=256, q=3329, sigma=2.5, iters_mul=2000, iters_ed=500):
    np.random.seed(0)

    bits_256 = np.random.randint(0, 2, n).astype(int)

    rlwe_s = RingLWE_schoolbook(n, q, sigma)
    rlwe_n = RingLWE_ntt(n, q, sigma)

    s_key_s = rlwe_s.generate_shared_secret()
    s_key_n = rlwe_n.generate_shared_secret()

    a_s = rlwe_s.generate_uniform()
    a_n = rlwe_n.generate_uniform()

    # --- multiplication only ---
    t_total_s, t_avg_s = bench_time(lambda: rlwe_s.multiply_polynomials(a_s, s_key_s), iters=iters_mul)
    t_total_n, t_avg_n = bench_time(lambda: rlwe_n.multiply_polynomials(a_n, s_key_n), iters=iters_mul)

    mem_cur_s, mem_peak_s = bench_memory(lambda: rlwe_s.multiply_polynomials(a_s, s_key_s), iters=200)
    mem_cur_n, mem_peak_n = bench_memory(lambda: rlwe_n.multiply_polynomials(a_n, s_key_n), iters=200)

    # --- encrypt + decrypt ---
    def ed_schoolbook():
        ct = rlwe_s.encrypt(s_key_s, bits_256)
        rlwe_s.decrypt(s_key_s, ct)

    def ed_ntt():
        ct = rlwe_n.encrypt(s_key_n, bits_256)
        rlwe_n.decrypt(s_key_n, ct)

    t_total_ed_s, t_avg_ed_s = bench_time(ed_schoolbook, iters=iters_ed, warmup=50)
    t_total_ed_n, t_avg_ed_n = bench_time(ed_ntt, iters=iters_ed, warmup=50)

    print("=== Multiplication only ===")
    print("schoolbook total:", t_total_s, "avg:", t_avg_s)
    print("ntt       total:", t_total_n, "avg:", t_avg_n)
    print()
    print("=== Encrypt+Decrypt ===")
    print("schoolbook total:", t_total_ed_s, "avg:", t_avg_ed_s)
    print("ntt       total:", t_total_ed_n, "avg:", t_avg_ed_n)
    print()
    print("=== Peak memory (tracemalloc) ===")
    print("schoolbook peak bytes:", mem_peak_s)
    print("ntt       peak bytes:", mem_peak_n)

# call it
run_all()