# ntt_kyber_py.py
# Kyber-style NTT in Python (q=3329, n=256), ring Z_q[x]/(x^256 + 1)

from __future__ import annotations
import numpy as np

KYBER_N = 256
KYBER_Q = 3329

# QINV = -q^{-1} mod 2^16 (Kyber uses this in Montgomery reduction)
QINV = 62209  # 3329 * 62209 ≡ -1 (mod 2^16)

# Montgomery R = 2^16, MONT = R mod q
MONT = 2285

# zetas from pq-crystals/kyber (int16_t zetas[128])
ZETAS = np.array([
  -1044,  -758,  -359, -1517,  1493,  1422,   287,   202,
   -171,   622,  1577,   182,   962, -1202, -1474,  1468,
    573, -1325,   264,   383,  -829,  1458, -1602,  -130,
   -681,  1017,   732,   608, -1542,   411,  -205, -1571,
   1223,   652,  -552,  1015, -1293,  1491,  -282, -1544,
    516,    -8,  -320,  -666, -1618, -1162,   126,  1469,
   -853,   -90,  -271,   830,   107, -1421,  -247,  -951,
   -398,   961, -1508,  -725,   448, -1065,   677, -1275,
  -1103,   430,   555,   843, -1251,   871,  1550,   105,
    422,   587,   177,  -235,  -291,  -460,  1574,  1653,
   -246,   778,  1159,  -147,  -777,  1483,  -602,  1119,
  -1590,   644,  -872,   349,   418,   329,  -156,   -75,
    817,  1097,   603,   610,  1322, -1285, -1465,   384,
  -1215,  -136,  1218, -1335,  -874,   220, -1187, -1659,
  -1185, -1530, -1278,   794, -1510,  -854,  -870,   478,
   -108,  -308,   996,   991,   958, -1460,  1522,  1628
], dtype=np.int32)

# ------------------------------------------------------------
# Reduction helpers
# ------------------------------------------------------------

def montgomery_reduce(a: np.ndarray | int) -> np.ndarray | int:
    """
    Montgomery reduction for q=3329 with R=2^16.
    Returns a * R^{-1} mod q.
    Works for scalar int or numpy array int32/int64.
    """
    # t = (a * QINV) mod 2^16
    # u = (a - t*q) / 2^16
    if isinstance(a, np.ndarray):
        t = (a * QINV) & 0xFFFF
        u = (a - t * KYBER_Q) >> 16
        return u.astype(np.int32)
    else:
        t = (a * QINV) & 0xFFFF
        u = (a - t * KYBER_Q) >> 16
        return int(u)

def barrett_reduce(a: np.ndarray | int) -> np.ndarray | int:
    """
    Barrett reduction mod q (Kyber style).
    """
    # v = round(2^26 / q)
    v = 20159  # floor((1<<26)/3329 + 0.5) used in Kyber ref
    if isinstance(a, np.ndarray):
        t = (v * a + (1 << 25)) >> 26
        r = a - t * KYBER_Q
        return r.astype(np.int32)
    else:
        t = (v * a + (1 << 25)) >> 26
        return int(a - t * KYBER_Q)

def fqmul(a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray | int:
    """
    Multiply then Montgomery reduce (Kyber fqmul).
    """
    return montgomery_reduce((a * b).astype(np.int64) if isinstance(a, np.ndarray) else a * b)

# ------------------------------------------------------------
# Kyber NTT / invNTT
# ------------------------------------------------------------

def ntt(r: np.ndarray) -> np.ndarray:
    """
    In-place Kyber NTT.
    Input: standard order
    Output: bit-reversed order (Kyber convention)
    r must be length 256, dtype int32/int64.
    """
    if r.shape[0] != KYBER_N:
        raise ValueError("ntt expects length 256")
    r = r.astype(np.int32, copy=False)

    k = 1
    length = 128
    while length >= 2:
        for start in range(0, KYBER_N, 2 * length):
            zeta = int(ZETAS[k])
            k += 1

            a = r[start:start + length]
            b = r[start + length:start + 2 * length]

            t = fqmul(zeta, b)
            r[start:start + length] = a + t
            r[start + length:start + 2 * length] = a - t
        length //= 2

    # Keep values in a reasonable range mod q
    r[:] = barrett_reduce(r)  # Kyber reduces during invntt too; this helps avoid overflow
    return r

def invntt_tomont(r: np.ndarray) -> np.ndarray:
    """
    In-place inverse Kyber NTT + multiply by montgomery factor.
    Input: bit-reversed order
    Output: standard order
    """
    if r.shape[0] != KYBER_N:
        raise ValueError("invntt_tomont expects length 256")
    r = r.astype(np.int32, copy=False)

    k = 127
    length = 2
    while length <= 128:
        for start in range(0, KYBER_N, 2 * length):
            zeta = int(ZETAS[k])
            k -= 1

            a = r[start:start + length]
            b = r[start + length:start + 2 * length]

            t = a.copy()
            a_new = barrett_reduce(t + b)
            b_new = t - b
            b_new = fqmul(zeta, b_new)

            r[start:start + length] = a_new
            r[start + length:start + 2 * length] = b_new
        length *= 2

    # f = mont^2 / 128 (Kyber constant)
    f = 1441
    r[:] = fqmul(r, f)
    r[:] = barrett_reduce(r)
    return r

# ------------------------------------------------------------
# Polynomial multiplication in Z_q[x]/(x^256+1)
# ------------------------------------------------------------

def poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Negacyclic multiplication mod (x^256+1, q) using Kyber NTT domain.
    """
    a = np.array(a, dtype=np.int32, copy=True)
    b = np.array(b, dtype=np.int32, copy=True)
    if a.shape[0] != KYBER_N or b.shape[0] != KYBER_N:
        raise ValueError("poly_mul expects length-256 polynomials")

    # NTT both
    ntt(a)
    ntt(b)

    # Pointwise multiply (needs Montgomery reduction)
    c = fqmul(a, b).astype(np.int32)

    # Inverse NTT
    invntt_tomont(c)

    # Final mod q
    c %= KYBER_Q
    return c

# ------------------------------------------------------------
# Quick sanity test
# ------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    a = rng.integers(0, KYBER_Q, size=KYBER_N, dtype=np.int32)
    b = rng.integers(0, KYBER_Q, size=KYBER_N, dtype=np.int32)

    c = poly_mul(a, b)
    print("poly_mul computed. c[0:8] =", c[:8])