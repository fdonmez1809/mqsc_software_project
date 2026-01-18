import numpy as np
import matplotlib.pyplot as plt

# polynomial degrees
n = np.arange(1, 513)

# asymptotic complexities
schoolbook = n**2
ntt = n * np.log2(n)

# plot
plt.figure(figsize=(8, 5))
plt.plot(n, schoolbook, label=r"$O(n^2)$ (Schoolbook)")
plt.plot(n, ntt, label=r"$O(n \log n)$ (NTT)")

plt.xlabel("Polynomial degree n")
plt.ylabel("Relative operation count")
plt.title("Asymptotic Complexity Comparison\nSchoolbook vs NTT Polynomial Multiplication")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()