from pathlib import Path
p = Path("data/raw/potw2049a.jpg")
b = p.read_bytes()
print("Byte length:", len(b))
print("First 32 bytes:", b[:32])



from pathlib import Path
img = Path("data/raw/potw2049a.jpg").read_bytes()
Path("data/generated/potw2049a.bin").write_bytes(img)
print("Wrote data/generated/potw2049a.bin with", len(img), "bytes")
