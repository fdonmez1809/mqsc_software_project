import numpy as np
from pathlib import Path

# Read file as bytes
byte_data = Path("data/generated/potw2049a.bin").read_bytes()

# Convert to NumPy array of uint8
byte_array = np.frombuffer(byte_data, dtype=np.uint8)

# Convert bytes → bits
bit_array = np.unpackbits(byte_array)

print("Total bits:", bit_array.shape[0])
print("First 64 bits:", bit_array[:64])