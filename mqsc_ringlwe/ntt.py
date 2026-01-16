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


def NTT_multiplication(p1,p2, n_input=None, q_input=None):

    ### PREPERATION

    degree_product = (len(p1)-1) + (len(p2)-1)
    lenght_product = degree_product + 1
    power = 0

    # n is the first power of 2 that exceeds len(c)
    while 2**power <= lenght_product:
        power = power + 1

    if n_input != None:
        power = int(np.log2(n_input))

    # q smallest prime such that q=>n*len(p1)*len(p2) + 1
    # and q = k*n -1 for a certain k
    threshold = (2**power)*len(p1)*len(p2) + 1
    q = threshold

    if q_input == None:
        while True:
            q += 1
            if is_prime(q) and ((q - 1) % 2**power == 0):
                k = (q-1)//(2**power)
                break
    else:
        q = q_input
        k = (q-1)//(2**power)

    # r = primitive root of q
    r = primitive_root(q)
    
    # define omega as w = r^k mod q
    # and omega_hat is all omegas where exponent is a power of 2
    omega = []
    omega_hat = []

    omega_base = pow(r, k, q)

    for i in range(degree_product + 1):
        val = pow(omega_base, i, q)
        omega.append(val)

        if i > 0 and (i & (i - 1)) == 0:
            omega_hat.append(val)

    ### NTT PART

    n = 2**power

    if n_input != None:
        n = n_input

    # add zero until it has the lenght of n
    a = p1 + [0] * (n - len(p1))
    b = p2 + [0] * (n - len(p2))

    # omega_hat_k list like in the lecture
    # omega_hat_1 is the top root, omega_hat_last is the bottom root
    omega_hat_k = []
    for level in range(power):
        omega_hat_k.append(pow(omega_base, 2**level, q))

    # split into u1 and u2 then recombine
    # forward NTT
    A = NTT_splitting(a, 0, omega_hat_k, q)
    B = NTT_splitting(b, 0, omega_hat_k, q)

    # pointwise multiplication in NTT form (after inverse they become their normal forms)
    C = [(A[i] * B[i]) % q for i in range(n)]

    # inverse NTT
    inv_omega_hat_k = [pow(w, q-2, q) for w in omega_hat_k]


    # inverse NTT
    c_time = inverse_NTT_splitting(C, 0, inv_omega_hat_k, q)

    # normalization process (divide by n modulo q)
    n_inv = pow(n, q-2, q)
    for i in range(n):
        c_time[i] = (c_time[i] * n_inv) % q

    result = c_time[:lenght_product]

    if n_input != None:
        result = c_time[:n_input]

    return result