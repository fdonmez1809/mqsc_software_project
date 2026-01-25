# mqsc_ringlwe/ringlwe_ntt_optimization.py
# ------------------------------------------------------------
# Optimized (vectorized) NTT-based polynomial multiply
# + a toy Ring-LWE wrapper that matches a common API style.
#
# Key point for your project:
# - If you want multiplication in the ring Z_q[x]/(x^n + 1) (negacyclic),
#   an NTT needs a primitive 2n-th root of unity in Z_q, i.e. 2n | (q-1).
# - With the Kyber-like modulus q=3329, q-1=3328 is divisible by 256 but NOT by 512.
#   So for n=256 you can do a *cyclic* NTT (x^n - 1), but you cannot do a
#   *negacyclic* NTT (x^n + 1) unless you change q (e.g., q=12289 supports 2n | (q-1)
#   for n=256).
#
# This file therefore supports TWO modes automatically:
#   1) mode="negacyclic" if (q-1) % (2n) == 0  -> implements mod (x^n + 1)
#   2) mode="cyclic"     if (q-1) % n == 0     -> implements mod (x^n - 1)
#
# For speed in Python, the butterflies are vectorized with NumPy reshapes.
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


def _factorize(n: int) -> list[int]:
    """Prime factorization (distinct primes) of n."""
    factors = []
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            factors.append(p)
            while x % p == 0:
                x //= p
        p += 1 if p == 2 else 2
    if x > 1:
        factors.append(x)
    return factors


def _find_generator_mod_prime(q: int) -> int:
    """
    Find a primitive generator g of multiplicative group Z_q^* (q prime).
    """
    phi = q - 1
    primes = _factorize(phi)
    for g in range(2, q):
        ok = True
        for p in primes:
            if pow(g, phi // p, q) == 1:
                ok = False
                break
        if ok:
            return g
    raise ValueError(f"No generator found for q={q} (unexpected if prime).")


def _bit_reverse_indices(n: int) -> np.ndarray:
    """Return bit-reversal permutation indices for n (power of two)."""
    logn = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x = i
        r = 0
        for _ in range(logn):
            r = (r << 1) | (x & 1)
            x >>= 1
        rev[i] = r
    return rev


@dataclass
class NTTPlan:
    n: int
    q: int
    mode: str  # "cyclic" or "negacyclic"

    omega: int
    omega_inv: int
    n_inv: int

    # Only used for negacyclic (x^n + 1)
    psi_pows: Optional[np.ndarray]
    psi_pows_inv: Optional[np.ndarray]

    rev: np.ndarray
    stage_roots: list[np.ndarray]
    stage_roots_inv: list[np.ndarray]


def make_ntt_plan(n: int, q: int, mode: str = "auto") -> NTTPlan:
    """
    Build an NTT plan. mode ∈ {"auto", "negacyclic", "cyclic"}.

    - If (q-1) % (2n) == 0: build a NEGACYCLIC plan for Z_q[x]/(x^n + 1)
      using the standard "twist" by powers of psi (a primitive 2n-th root).

    - Else if (q-1) % n == 0: build a CYCLIC plan for Z_q[x]/(x^n - 1).

    Requirements:
    - q should be prime (we use Fermat inverses and assume a generator exists)
    - n must be power of two
    """
    if n & (n - 1) != 0:
        raise ValueError("NTT requires n to be a power of two.")

    # Decide plan mode
    mode_req = (mode or "auto").lower()
    if mode_req not in {"auto", "negacyclic", "cyclic"}:
        raise ValueError(f"Invalid mode={mode!r}. Use 'auto', 'negacyclic', or 'cyclic'.")

    # Auto-prefer negacyclic when possible, otherwise cyclic.
    if mode_req == "auto":
        if (q - 1) % (2 * n) == 0:
            mode = "negacyclic"
        elif (q - 1) % n == 0:
            mode = "cyclic"
        else:
            raise ValueError(
                f"Need n | (q-1) for cyclic NTT or (2n) | (q-1) for negacyclic NTT. "
                f"Here q-1={q-1} not divisible by n={n} nor 2n={2*n}."
            )
    elif mode_req == "negacyclic":
        if (q - 1) % (2 * n) != 0:
            raise ValueError(
                f"Requested negacyclic NTT but (q-1) is not divisible by 2n. "
                f"q-1={q-1}, 2n={2*n}."
            )
        mode = "negacyclic"
    else:  # mode_req == "cyclic"
        if (q - 1) % n != 0:
            raise ValueError(
                f"Requested cyclic NTT but (q-1) is not divisible by n. "
                f"q-1={q-1}, n={n}."
            )
        mode = "cyclic"

    g = _find_generator_mod_prime(q)

    psi_pows: Optional[np.ndarray] = None
    psi_pows_inv: Optional[np.ndarray] = None

    if mode == "negacyclic":
        # psi is a primitive 2n-th root of unity
        psi = pow(g, (q - 1) // (2 * n), q)
        omega = (psi * psi) % q  # omega = psi^2 is primitive n-th root
        psi_inv = pow(psi, q - 2, q)

        psi_pows = np.empty(n, dtype=np.int64)
        psi_pows_inv = np.empty(n, dtype=np.int64)
        w = 1
        w_inv = 1
        for i in range(n):
            psi_pows[i] = w
            psi_pows_inv[i] = w_inv
            w = (w * psi) % q
            w_inv = (w_inv * psi_inv) % q

    else:
        # cyclic: omega is a primitive n-th root of unity
        omega = pow(g, (q - 1) // n, q)

    omega_inv = pow(omega, q - 2, q)
    n_inv = pow(n, q - 2, q)
    rev = _bit_reverse_indices(n)

    # Precompute roots for each stage (DIT Cooley-Tukey).
    logn = n.bit_length() - 1
    stage_roots: list[np.ndarray] = []
    stage_roots_inv: list[np.ndarray] = []

    for s in range(1, logn + 1):
        m = 1 << s
        half = m >> 1

        w_m = pow(omega, n // m, q)
        w_m_inv = pow(omega_inv, n // m, q)

        roots = np.empty(half, dtype=np.int64)
        roots_inv = np.empty(half, dtype=np.int64)

        w = 1
        w_inv = 1
        for j in range(half):
            roots[j] = w
            roots_inv[j] = w_inv
            w = (w * w_m) % q
            w_inv = (w_inv * w_m_inv) % q

        stage_roots.append(roots)
        stage_roots_inv.append(roots_inv)

    return NTTPlan(
        n=n,
        q=q,
        mode=mode,
        omega=omega,
        omega_inv=omega_inv,
        n_inv=n_inv,
        psi_pows=psi_pows,
        psi_pows_inv=psi_pows_inv,
        rev=rev,
        stage_roots=stage_roots,
        stage_roots_inv=stage_roots_inv,
    )


def ntt_inplace(a: np.ndarray, plan: NTTPlan) -> None:
    """In-place forward NTT (DIT), vectorized per stage."""
    n = plan.n
    q = plan.q

    a[:] = a[plan.rev]

    logn = n.bit_length() - 1
    for s in range(1, logn + 1):
        m = 1 << s
        half = m >> 1
        roots = plan.stage_roots[s - 1].astype(np.int64, copy=False)

        blocks = a.reshape(-1, m)
        u = blocks[:, :half]
        v = blocks[:, half:]

        t = (v * roots) % q
        blocks[:, :half] = (u + t) % q
        blocks[:, half:] = (u - t) % q


def intt_inplace(a: np.ndarray, plan: NTTPlan) -> None:
    """In-place inverse NTT (DIT with inverse roots), vectorized per stage."""
    n = plan.n
    q = plan.q

    a[:] = a[plan.rev]

    logn = n.bit_length() - 1
    for s in range(1, logn + 1):
        m = 1 << s
        half = m >> 1
        roots_inv = plan.stage_roots_inv[s - 1].astype(np.int64, copy=False)

        blocks = a.reshape(-1, m)
        u = blocks[:, :half]
        v = blocks[:, half:]

        t = (v * roots_inv) % q
        blocks[:, :half] = (u + t) % q
        blocks[:, half:] = (u - t) % q

    a[:] = (a * plan.n_inv) % q


def poly_mul_ntt(a: np.ndarray, b: np.ndarray, plan: NTTPlan) -> np.ndarray:
    """Multiply polynomials using the plan's mode (cyclic or negacyclic)."""
    q = plan.q

    fa = np.array(a, dtype=np.int64, copy=True) % q
    fb = np.array(b, dtype=np.int64, copy=True) % q

    if plan.mode == "negacyclic":
        # Twist: multiply by powers of psi
        fa = (fa * plan.psi_pows) % q  # type: ignore[arg-type]
        fb = (fb * plan.psi_pows) % q  # type: ignore[arg-type]

    ntt_inplace(fa, plan)
    ntt_inplace(fb, plan)

    fa = (fa * fb) % q

    intt_inplace(fa, plan)

    if plan.mode == "negacyclic":
        # Untwist
        fa = (fa * plan.psi_pows_inv) % q  # type: ignore[arg-type]

    return fa


# ------------------------------------------------------------
# Toy Ring-LWE wrapper (API-friendly for your benchmark scripts)
# ------------------------------------------------------------

class RingLWE:
    """
    A compact Ring-LWE-like toy scheme for benchmarking.
    This is not a production cryptosystem.

    Ring: Z_q[x]/(x^n ± 1)  (plan.mode selects cyclic/negacyclic)
    """

    def __init__(self, n: int = 256, q: int = 3329, sigma: float = 2.5, seed: int = 1):
        self.n = int(n)
        self.q = int(q)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

        self.plan = make_ntt_plan(self.n, self.q, mode="auto")

        # Keep a readable name for reporting/debugging
        self.ring_mode = self.plan.mode

        # Precompute encoding scale for bit messages (0/1)
        self._half = self.q // 2

    # -------------------------
    # Sampling helpers
    # -------------------------

    def _modq(self, x: np.ndarray) -> np.ndarray:
        return np.array(x, dtype=np.int64, copy=False) % self.q

    def sample_uniform(self) -> np.ndarray:
        return self.rng.integers(0, self.q, size=self.n, dtype=np.int64)

    def sample_small(self) -> np.ndarray:
        # discrete Gaussian-ish: round normal, then mod q
        e = np.rint(self.rng.normal(0.0, self.sigma, size=self.n)).astype(np.int64)
        return e % self.q

    # -------------------------
    # Core polynomial multiply
    # -------------------------

    def multiply_polynomials(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = self._modq(a)
        b = self._modq(b)
        return poly_mul_ntt(a, b, self.plan)

    # -------------------------
    # Key / Encrypt / Decrypt
    # -------------------------

    def generate_shared_secret(self) -> np.ndarray:
        # secret s (small)
        return self.sample_small()

    def keygen(self) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Returns (sk, pk) where pk=(a, b) with b = a*s + e  (mod q).
        """
        a = self.sample_uniform()
        s = self.sample_small()
        e = self.sample_small()
        b = (self.multiply_polynomials(a, s) + e) % self.q
        return s, (a, b)

    def encode_message(self, m_bits: np.ndarray) -> np.ndarray:
        """
        Map bits {0,1}^n into Z_q^n by scaling 1 -> q/2.
        """
        m = np.array(m_bits, dtype=np.int64)
        if m.shape[0] != self.n:
            raise ValueError(f"Message length must be n={self.n}")
        return (m % 2) * self._half

    def decode_message(self, m_coeffs: np.ndarray) -> np.ndarray:
        """
        Decode Z_q^n back to bits by threshold around q/4..3q/4.
        """
        x = np.array(m_coeffs, dtype=np.int64) % self.q
        # centered representative in (-q/2, q/2]
        centered = ((x + self.q // 2) % self.q) - self.q // 2
        # if close to q/2 -> 1, else 0
        return (np.abs(centered) > (self.q // 4)).astype(int)

    def encrypt(
        self,
        pk_or_a: np.ndarray,
        m_bits: np.ndarray,
        pk_b: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encrypt interface supports either:
          encrypt(pk=(a,b), m_bits)  by passing pk_or_a as tuple in your code, OR
          encrypt(a, m_bits, b)

        To stay compatible with many student codes, we accept:
          - if pk_b is None and pk_or_a is a tuple/list of length 2 -> treat as (a,b)
          - else treat pk_or_a=a and pk_b=b
        """
        if pk_b is None and isinstance(pk_or_a, (tuple, list)) and len(pk_or_a) == 2:
            a, b = pk_or_a
        else:
            a, b = pk_or_a, pk_b
            if b is None:
                raise ValueError("encrypt needs (a,b) or (a, b=...)")

        a = self._modq(a)
        b = self._modq(b)
        m_enc = self.encode_message(m_bits)

        r = self.sample_small()
        e1 = self.sample_small()
        e2 = self.sample_small()

        u = (self.multiply_polynomials(a, r) + e1) % self.q
        v = (self.multiply_polynomials(b, r) + e2 + m_enc) % self.q
        return u, v

    def decrypt(self, sk: np.ndarray, ct: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        u, v = ct
        u = self._modq(u)
        v = self._modq(v)
        s = self._modq(sk)

        # m' = v - u*s
        us = self.multiply_polynomials(u, s)
        m_coeffs = (v - us) % self.q
        return self.decode_message(m_coeffs)


# ------------------------------------------------------------
# Minimal self-test (optional)
# ------------------------------------------------------------
if __name__ == "__main__":
    n = 256
    q = 3329
    rlwe = RingLWE(n=n, q=q, sigma=2.5, seed=1)

    sk, pk = rlwe.keygen()
    msg = rlwe.rng.integers(0, 2, size=n, dtype=int)
    ct = rlwe.encrypt(pk, msg)
    dec = rlwe.decrypt(sk, ct)

    print("mode:", rlwe.ring_mode, "| decrypt success:", float(np.mean(dec == msg)))