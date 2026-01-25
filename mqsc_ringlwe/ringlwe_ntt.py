import numpy as np

from .ringlwe_ntt_optimization import make_ntt_plan, poly_mul_ntt


class RingLWE:
    """RingLWE wrapper using the NTT backend.

    This class supports arbitrary power-of-two `n` values **as long as** an NTT plan
    can be constructed for `(n, q)`:
      - cyclic NTT if n | (q-1)
      - negacyclic NTT if 2n | (q-1)

    The ring mode is determined by the plan (plan.mode).
    """

    def __init__(self, n: int, q: int, sigma: float):
        self.n = int(n)
        self.q = int(q)
        self.sigma = float(sigma)

        try:
            # mode is auto by default (see make_ntt_plan)
            self.plan = make_ntt_plan(self.n, self.q, mode="auto")
        except Exception as e:
            raise ValueError(f"Cannot create NTT plan for (n,q)=({self.n},{self.q}): {e}")

    def add_polynomials(self, p1, p2):
        return (np.asarray(p1, dtype=np.int64) + np.asarray(p2, dtype=np.int64)) % self.q

    def subtract_polynomials(self, p1, p2):
        return (np.asarray(p1, dtype=np.int64) - np.asarray(p2, dtype=np.int64)) % self.q

    def multiply_polynomials(self, p1, p2):
        p1 = np.asarray(p1, dtype=np.int64) % self.q
        p2 = np.asarray(p2, dtype=np.int64) % self.q
        if p1.shape[0] != self.n or p2.shape[0] != self.n:
            raise ValueError(
                f"Polynomial length mismatch: expected n={self.n}, got {p1.shape[0]} and {p2.shape[0]}"
            )

        result = poly_mul_ntt(p1, p2, self.plan)
        return np.asarray(result, dtype=np.int64) % self.q

    def generate_error(self):
        error = np.random.normal(0.0, self.sigma, self.n)
        return np.rint(error).astype(np.int64) % self.q

    def generate_uniform(self):
        return np.random.randint(0, self.q, self.n, dtype=np.int64)

    def generate_shared_secret(self):
        return self.generate_error()

    def encrypt(self, s, message_bits):
        a = self.generate_uniform()
        e = self.generate_error()
        message_poly = np.asarray(message_bits, dtype=np.int64) * (self.q // 2)

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
        q4 = self.q // 4
        q34 = 3 * self.q // 4
        for coefficient in np.asarray(message_with_noise, dtype=np.int64) % self.q:
            if q4 < coefficient < q34:
                decrypted_bits.append(1)
            else:
                decrypted_bits.append(0)
        return decrypted_bits