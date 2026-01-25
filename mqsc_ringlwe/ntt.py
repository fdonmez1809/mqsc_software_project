import numpy as np

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def primitive_root(q):
    theta = q-1
    prime_factors_theta = prime_factors(theta)
    for u in range(2, q):
        is_primitive = True
        for p in prime_factors_theta:
            if pow(u, theta // p, q) == 1:
                is_primitive = False
                break
        if is_primitive:
            return u

def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def NTT_splitting(u, level, omega_hat_k, q):
    m = len(u)

    # bottom of the tree
    if m == 1:
        return [u[0] % q]

    # split (even and odd indices)
    u1 = u[0::2]
    u2 = u[1::2]

    # go down the tree
    u1_prime = NTT_splitting(u1, level + 1, omega_hat_k, q)
    u2_prime = NTT_splitting(u2, level + 1, omega_hat_k, q)

    # recombine with omega_hat at this level (lecture butterfly)
    u_prime = [0] * m
    omega_hat = omega_hat_k[level]

    w = 1
    for i in range(m // 2):
        t = (w * u2_prime[i]) % q
        u_prime[i] = (u1_prime[i] + t) % q
        u_prime[i + m // 2] = (u1_prime[i] - t) % q
        w = (w * omega_hat) % q

    return u_prime



def inverse_NTT_splitting(u, level, inv_omega_hat_k, q):
    m = len(u)

    if m == 1:
        return [u[0] % q]

    u1 = u[0::2]
    u2 = u[1::2]

    u1_prime = inverse_NTT_splitting(u1, level + 1, inv_omega_hat_k, q)
    u2_prime = inverse_NTT_splitting(u2, level + 1, inv_omega_hat_k, q)

    u_prime = [0] * m
    omega_hat = inv_omega_hat_k[level]

    w = 1
    for i in range(m // 2):
        t = (w * u2_prime[i]) % q
        u_prime[i] = (u1_prime[i] + t) % q
        u_prime[i + m // 2] = (u1_prime[i] - t) % q
        w = (w * omega_hat) % q

    return u_prime


def find_q_for_n(n):
    """Find the smallest prime q > n^2 + 1 such that n divides (q-1)."""
    threshold = n**2 + 1
    q = threshold + 1
    while True:
        if is_prime(q) and (q - 1) % n == 0:
            return q
        q += 1

def NTT_multiplication(p1, p2, n, q=None):
    if q is None:
        q = find_q_for_n(n)
    # Assume n is a power of 2, q is prime, and n divides (q-1)
    power = int(np.log2(n))
    k = (q - 1) // n
    r = primitive_root(q)
    omega_base = pow(r, k, q)
    omega_hat_k = [pow(omega_base, 2**level, q) for level in range(power)]
    
    # Pad p1 and p2 to length n
    a = p1 + [0] * (n - len(p1))
    b = p2 + [0] * (n - len(p2))
    
    # Forward NTT
    A = NTT_splitting(a, 0, omega_hat_k, q)
    B = NTT_splitting(b, 0, omega_hat_k, q)
    C = [(A[i] * B[i]) % q for i in range(n)]
    
    # Inverse NTT
    inv_omega_hat_k = [pow(w, q - 2, q) for w in omega_hat_k]
    c_time = inverse_NTT_splitting(C, 0, inv_omega_hat_k, q)
    
    # Normalize by dividing by n modulo q
    n_inv = pow(n, q - 2, q)
    for i in range(n):
        c_time[i] = (c_time[i] * n_inv) % q
    
    return c_time