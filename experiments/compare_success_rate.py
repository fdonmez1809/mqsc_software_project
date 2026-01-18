import numpy as np
from pathlib import Path

from mqsc_ringlwe.ringlwe_schoolbook import RingLWE as RingLWE_Schoolbook
from mqsc_ringlwe.ringlwe_ntt import RingLWE as RingLWE_NTT


def file_to_bits(path):
    byte_data = Path(path).read_bytes()
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array)
    return bit_array.astype(int)


def split_bits_to_blocks(bit_array, n):
    total_bits = bit_array.shape[0]
    n_blocks = total_bits // n
    trimmed = bit_array[: n_blocks * n]
    blocks = trimmed.reshape(n_blocks, n)
    return blocks


def evaluate_rlwe(rlwe, blocks, trials_limit=None, seed=0):
    rng = np.random.default_rng(seed)

    if trials_limit is not None:
        if trials_limit < blocks.shape[0]:
            idx = rng.choice(blocks.shape[0], size=trials_limit, replace=False)
            blocks = blocks[idx]

    key = rlwe.generate_shared_secret()

    total_blocks = blocks.shape[0]
    total_bits = total_blocks * rlwe.n

    correct_blocks = 0
    correct_bits = 0

    mismatch_examples = []

    for i in range(total_blocks):
        m = blocks[i]
        a, b = rlwe.encrypt(key, m)
        dec = rlwe.decrypt(key, (a, b))
        dec = np.array(dec, dtype=int)

        matches = (dec == m)
        bit_ok = int(matches.sum())
        correct_bits += bit_ok

        if bit_ok == rlwe.n:
            correct_blocks += 1
        else:
            if len(mismatch_examples) < 3:
                mismatch_positions = np.where(~matches)[0].tolist()
                mismatch_examples.append(
                    {
                        "block_index": int(i),
                        "wrong_bits": int(rlwe.n - bit_ok),
                        "positions": mismatch_positions[:20],
                    }
                )

    block_success_rate = correct_blocks / total_blocks if total_blocks > 0 else 0.0
    bit_accuracy = correct_bits / total_bits if total_bits > 0 else 0.0

    return {
        "total_blocks": int(total_blocks),
        "total_bits": int(total_bits),
        "correct_blocks": int(correct_blocks),
        "correct_bits": int(correct_bits),
        "block_success_rate": float(block_success_rate),
        "bit_accuracy": float(bit_accuracy),
        "mismatch_examples": mismatch_examples,
    }


def main():
    # change this path to your file
    file_path = "data/raw/potw2049a.jpg"
    # file_path = "data/generated/potw2049a.bin"

    n = 256
    q = 3329
    sigma = 2.5

    bits = file_to_bits(file_path)
    blocks = split_bits_to_blocks(bits, n)

    print("File:", file_path)
    print("Total bits:", bits.shape[0])
    print("Blocks used:", blocks.shape[0], "(each block is n =", n, "bits)")
    print()

    rlwe_schoolbook = RingLWE_Schoolbook(n, q, sigma)
    rlwe_ntt = RingLWE_NTT(n, q, sigma)

    # If you want faster testing, set trials_limit to something like 200 or 1000
    trials_limit = None

    schoolbook_res = evaluate_rlwe(rlwe_schoolbook, blocks, trials_limit=trials_limit, seed=1)
    ntt_res = evaluate_rlwe(rlwe_ntt, blocks, trials_limit=trials_limit, seed=1)

    print("=== Schoolbook ===")
    print("total_blocks:", schoolbook_res["total_blocks"])
    print("block_success_rate:", schoolbook_res["block_success_rate"])
    print("bit_accuracy:", schoolbook_res["bit_accuracy"])
    print("mismatch_examples:", schoolbook_res["mismatch_examples"])
    print()

    print("=== NTT ===")
    print("total_blocks:", ntt_res["total_blocks"])
    print("block_success_rate:", ntt_res["block_success_rate"])
    print("bit_accuracy:", ntt_res["bit_accuracy"])
    print("mismatch_examples:", ntt_res["mismatch_examples"])
    print()

    print("=== Comparison (NTT - Schoolbook) ===")
    print("block_success_rate diff:", ntt_res["block_success_rate"] - schoolbook_res["block_success_rate"])
    print("bit_accuracy diff:", ntt_res["bit_accuracy"] - schoolbook_res["bit_accuracy"])


if __name__ == "__main__":
    main()