import numpy as np

class RingLWE:
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
        
        # 1. Remove the mask: b - a*s = (a*s + e + m) - a*s = e + m
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
    
n = 256        # ring dimension
q = 3329       # modulus
sigma = 2.5    # noise standard deviation


bits_256 = np.array([
    0,1,1,1,0,1,0,0, 0,1,1,0,0,1,0,1, 0,1,1,1,1,0,0,0, 0,1,1,1,0,1,0,0,
    0,0,1,0,0,0,0,0, 0,1,1,0,1,0,0,1, 0,1,1,0,1,1,1,0, 0,1,1,1,0,0,0,0,
    0,1,1,1,0,1,0,1, 0,1,1,0,1,1,0,0, 0,1,1,0,0,1,0,1, 0,1,1,1,0,1,0,0,
    0,1,1,1,1,0,0,1, 0,1,1,0,1,0,0,1, 0,1,1,1,0,1,0,1, 0,0,1,0,0,0,1,1,
    1,0,0,1,1,0,1,0, 0,1,1,0,1,1,0,1, 0,1,1,1,0,0,1,0, 0,1,1,0,0,1,1,1,
    0,1,1,1,0,1,0,0, 0,1,1,0,1,1,1,0, 0,1,1,0,0,1,0,0, 0,1,1,1,0,1,0,1,
    0,0,1,0,1,0,1,1, 0,1,1,0,1,0,0,1, 0,1,1,1,1,0,0,0, 0,1,1,0,0,1,0,1,
    0,1,1,1,0,1,1,0, 0,1,1,0,1,0,1,0, 0,1,1,1,0,0,1,0, 0,1,1,0,0,0,1,1
], dtype=int)

rlwe = RingLWE(n, q, sigma)
key = rlwe.generate_shared_secret()
encrypted = rlwe.encrypt(key,bits_256)
print(encrypted)
decrypted = rlwe.decrypt(key,encrypted)
print(decrypted)
    # p1= [5,3,4]
    # p2= [3,-8,4]
    # result = multiply_polynomials(p1,p2,3)
    # print(result)