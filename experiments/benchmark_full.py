# experiments/benchmark_full.py
# Benchmark: Schoolbook vs NTT (RingLWE)
#
# Measures:
# - Time: polynomial multiplication
# - Time: encrypt+decrypt
# - Memory: peak tracemalloc (KiB)
# - Correctness: decryption success rate + bit accuracy
# - Ring consistency: mismatch count between schoolbook vs NTT multiplication
# - Scalability: performance vs n (skips unsupported n automatically)
#
# Run:
#   python3 -m experiments.benchmark_full
# or:
#   python3 experiments/benchmark_full.py
#
# Optional:
#   python3 -m experiments.benchmark_full --n 64 128 256 512
#   python3 -m experiments.benchmark_full --mul-trials 400 --ed-trials 100
#   python3 -m experiments.benchmark_full --q 12289 --sigma 2.5 --seed 1

import os
import sys
import time
import tracemalloc
import argparse
from statistics import median

import numpy as np


# --- make imports work whether run as module or script ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# --- Schoolbook import and adapters ---
from mqsc_ringlwe.ringlwe_schoolbook import RingLWE as RingLWE_Schoolbook


# ----------------------------
# Adapter: force no-NumPy schoolbook path
# ----------------------------
class RingLWE_Schoolbook_NoNumPy(RingLWE_Schoolbook):
    """Adapter: force the no-NumPy schoolbook path if available.

    Uses:
      - encrypt_no_numPy / decrypt_no_numPy if they exist
      - multiply_polynomials_no_numPy if it exists
    """

    def encrypt(self, s, message_bits):
        if hasattr(self, "encrypt_no_numPy") and callable(getattr(self, "encrypt_no_numPy")):
            return self.encrypt_no_numPy(s, message_bits)
        return super().encrypt(s, message_bits)

    def decrypt(self, s, ciphertext):
        if hasattr(self, "decrypt_no_numPy") and callable(getattr(self, "decrypt_no_numPy")):
            return self.decrypt_no_numPy(s, ciphertext)
        return super().decrypt(s, ciphertext)

    def multiply_polynomials(self, a, b):
        if hasattr(self, "multiply_polynomials_no_numPy") and callable(getattr(self, "multiply_polynomials_no_numPy")):
            return self.multiply_polynomials_no_numPy(a, b)
        return super().multiply_polynomials(a, b)
# Kyber-style negacyclic NTT (q=3329, n=256) implemented in Python
# from mqsc_ringlwe.ntt_kyber_py import poly_mul as kyber_poly_mul
from mqsc_ringlwe.ringlwe_ntt_optimization import RingLWE as RingLWE_NTT


# Alias used by the benchmark below
# RingLWE_NTT = RingLWE_NewNTT


def _now_ns() -> int:
    return time.perf_counter_ns()


def _time_median_s(fn, repeat: int = 15, warmup: int = 3) -> float:
    """Return median runtime in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = _now_ns()
        fn()
        t1 = _now_ns()
        times.append((t1 - t0) / 1e9)
    return median(times)


def _time_quantiles_s(fn, repeat: int = 21, warmup: int = 5):
    """Return (median, p25, p75) seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = _now_ns()
        fn()
        t1 = _now_ns()
        times.append((t1 - t0) / 1e9)
    times = sorted(times)
    med = times[len(times) // 2]
    p25 = times[int(0.25 * (len(times) - 1))]
    p75 = times[int(0.75 * (len(times) - 1))]
    return med, p25, p75


def _peak_mem_kib_during(fn) -> float:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


# ----------------------------
# API-compat wrapper helpers
# ----------------------------
def _has_keypair_api(rlwe) -> bool:
    return hasattr(rlwe, "keygen") and callable(getattr(rlwe, "keygen"))


def _enc_dec_once(rlwe, msg: np.ndarray):
    """
    Runs one encrypt+decrypt using whichever API the RLWE object provides.

    Supports:
      A) keypair-style: (sk, pk)=keygen(); ct=encrypt(pk,msg); dec=decrypt(sk,ct)
      B) shared-secret-style: s=generate_shared_secret(); ct=encrypt(s,msg); dec=decrypt(s,ct)
    """
    n = rlwe.n
    if msg.shape[0] != n:
        raise ValueError("msg length mismatch")

    if _has_keypair_api(rlwe):
        sk, pk = rlwe.keygen()
        ct = rlwe.encrypt(pk, msg)
        dec = rlwe.decrypt(sk, ct)
        return dec

    if hasattr(rlwe, "generate_shared_secret"):
        s = rlwe.generate_shared_secret()
    elif hasattr(rlwe, "generate_error"):
        s = rlwe.generate_error()
    else:
        raise AttributeError("No key generator in RingLWE class.")

    ct = rlwe.encrypt(s, msg)
    dec = rlwe.decrypt(s, ct)
    return dec


def _supports_n(rlwe) -> (bool, str):
    """Smoke-test: multiplication + enc/dec once."""
    try:
        n = rlwe.n
        q = rlwe.q
        a = np.random.randint(0, q, size=n, dtype=np.int64)
        b = np.random.randint(0, q, size=n, dtype=np.int64)
        _ = rlwe.multiply_polynomials(a, b)

        msg = np.random.randint(0, 2, size=n, dtype=int)
        _ = _enc_dec_once(rlwe, msg)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _bench_mul_only(rlwe, trials: int, repeat: int, warmup: int, seed: int):
    n, q = rlwe.n, rlwe.q
    rng = np.random.default_rng(seed)

    a_list = rng.integers(0, q, size=(trials, n), dtype=np.int64)
    b_list = rng.integers(0, q, size=(trials, n), dtype=np.int64)

    idx = 0

    def run():
        nonlocal idx
        _ = rlwe.multiply_polynomials(a_list[idx], b_list[idx])
        idx = (idx + 1) % trials

    # time (median + IQR)
    med_s, p25_s, p75_s = _time_quantiles_s(run, repeat=repeat, warmup=warmup)

    # peak memory during the same workload shape
    def mem_run():
        for _ in range(min(trials, 200)):  # cap to keep mem test reasonable
            run()

    peak_kib = _peak_mem_kib_during(mem_run)

    return {
        "mul_us_med": med_s * 1e6,
        "mul_us_p25": p25_s * 1e6,
        "mul_us_p75": p75_s * 1e6,
        "mul_peak_kib": peak_kib,
    }


def _bench_end_to_end(rlwe, trials: int, repeat: int, warmup: int, seed: int):
    n = rlwe.n
    rng = np.random.default_rng(seed)

    # message blocks (avoid counting keygen inside enc/dec timing as much as possible)
    msgs = rng.integers(0, 2, size=(trials, n), dtype=int)

    # Prepare keys for the runs
    if _has_keypair_api(rlwe):
        keypairs = [rlwe.keygen() for _ in range(trials)]  # list of (sk, pk)
    else:
        if hasattr(rlwe, "generate_shared_secret"):
            keys = [rlwe.generate_shared_secret() for _ in range(trials)]
        elif hasattr(rlwe, "generate_error"):
            keys = [rlwe.generate_error() for _ in range(trials)]
        else:
            raise AttributeError("No key generator in RingLWE class.")
        keypairs = keys  # store as keys for shared-secret mode

    idx = 0

    def run_one():
        nonlocal idx
        m = msgs[idx]
        if _has_keypair_api(rlwe):
            sk, pk = keypairs[idx]
            ct = rlwe.encrypt(pk, m)
            _ = rlwe.decrypt(sk, ct)
        else:
            s = keypairs[idx]
            ct = rlwe.encrypt(s, m)
            _ = rlwe.decrypt(s, ct)
        idx = (idx + 1) % trials

    # time per block (median + IQR)
    med_s, p25_s, p75_s = _time_quantiles_s(run_one, repeat=repeat, warmup=warmup)

    # memory peak during a small batch of enc/dec operations
    def mem_run():
        for _ in range(min(trials, 100)):
            run_one()

    peak_kib = _peak_mem_kib_during(mem_run)

    # correctness KPI: decryption success over all trials
    correct_blocks = 0
    correct_bits = 0
    mismatch_examples = []

    for i in range(trials):
        m = msgs[i]
        if _has_keypair_api(rlwe):
            sk, pk = keypairs[i]
            ct = rlwe.encrypt(pk, m)
            dec = rlwe.decrypt(sk, ct)
        else:
            s = keypairs[i]
            ct = rlwe.encrypt(s, m)
            dec = rlwe.decrypt(s, ct)

        dec = np.array(dec, dtype=int)

        matches = (dec == m)
        bit_ok = int(matches.sum())
        correct_bits += bit_ok
        if bit_ok == n:
            correct_blocks += 1
        else:
            if len(mismatch_examples) < 3:
                wrong_pos = np.where(~matches)[0].tolist()
                mismatch_examples.append(
                    {
                        "trial": int(i),
                        "wrong_bits": int(n - bit_ok),
                        "positions": wrong_pos[:20],
                    }
                )

    block_success = correct_blocks / trials if trials else 0.0
    bit_acc = correct_bits / (trials * n) if trials else 0.0

    # throughput from median time
    blocks_per_s = (1.0 / med_s) if med_s > 0 else 0.0
    bits_per_s = blocks_per_s * n

    return {
        "ed_us_med": med_s * 1e6,
        "ed_us_p25": p25_s * 1e6,
        "ed_us_p75": p75_s * 1e6,
        "ed_peak_kib": peak_kib,
        "block_success_rate": block_success,
        "bit_accuracy": bit_acc,
        "throughput_blocks_s": blocks_per_s,
        "throughput_bits_s": bits_per_s,
        "mismatch_examples": mismatch_examples,
    }


def _bench_ring_consistency(n: int, q: int, trials: int, seed: int):
    """Compare schoolbook vs NTT polynomial multiplication for random polynomials."""
    rng = np.random.default_rng(seed)
    rlwe_s = RingLWE_Schoolbook(n=n, q=q, sigma=0.0)  # sigma irrelevant here

    # Kyber NTT adapter supports only (n,q)=(256,3329); other (n,q) pairs are skipped.
    try:
        rlwe_n = RingLWE_NTT(n=n, q=q, sigma=0.0)
    except Exception as e:
        return {"supported": False, "reason": f"NTT unsupported: {type(e).__name__}: {e}"}

    # If either doesn't support this n, fail gracefully
    ok_s, err_s = _supports_n(rlwe_s)
    ok_n, err_n = _supports_n(rlwe_n)
    if not ok_s:
        return {"supported": False, "reason": f"Schoolbook unsupported: {err_s}"}
    if not ok_n:
        return {"supported": False, "reason": f"NTT unsupported: {err_n}"}

    mismatches = 0
    first_examples = []
    max_abs_diff = 0

    for i in range(trials):
        a = rng.integers(0, q, size=n, dtype=np.int64)
        b = rng.integers(0, q, size=n, dtype=np.int64)

        ps = np.array(rlwe_s.multiply_polynomials(a, b), dtype=np.int64) % q
        pn = np.array(rlwe_n.multiply_polynomials(a, b), dtype=np.int64) % q

        if ps.shape != pn.shape or np.any(ps != pn):
            mismatches += 1
            diff = (pn - ps) % q
            diff = np.where(diff > q // 2, diff - q, diff)
            max_abs_diff = max(max_abs_diff, int(np.max(np.abs(diff))))
            if len(first_examples) < 3:
                bad = np.where(ps != pn)[0].tolist()
                first_examples.append({"trial": i, "num_bad_coeffs": len(bad), "positions": bad[:20]})

    return {
        "supported": True,
        "trials": trials,
        "mismatch_count": mismatches,
        "mismatch_rate": (mismatches / trials) if trials else 0.0,
        "max_abs_diff": max_abs_diff,
        "examples": first_examples,
    }


def _fmt_iqr(med, p25, p75) -> str:
    return f"{med:,.2f} [{p25:,.2f}, {p75:,.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=12289)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument(
        "--schoolbook-no-numpy",
        action="store_true",
        help="Use encrypt_no_numPy/decrypt_no_numPy (and multiply_polynomials_no_numPy if present) for Schoolbook.",
    )

    ap.add_argument("--n", type=int, nargs="*", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--mul-trials", type=int, default=300)
    ap.add_argument("--ed-trials", type=int, default=80)
    ap.add_argument("--consistency-trials", type=int, default=200)

    ap.add_argument("--repeat", type=int, default=21)
    ap.add_argument("--warmup", type=int, default=5)

    args = ap.parse_args()

    q = args.q
    sigma = args.sigma
    seed = args.seed

    print("Benchmark: Schoolbook vs NTT (RingLWE)")
    print(f"q={q}, sigma={sigma}, seed={seed}")
    print()

    header = (
        f"{'n':<6}"
        f"{'method':<12}"
        f"{'mul_us med[p25,p75]':>26}"
        f"{'enc+dec_us med[p25,p75]':>30}"
        f"{'peak_mem(KiB)':>15}"
        f"{'dec_success':>12}"
        f"{'bit_acc':>10}"
        f"{'blk/s':>10}"
    )
    print(header)
    print("-" * len(header))

    for n in args.n:
        q_n = q

        SchoolbookClass = RingLWE_Schoolbook_NoNumPy if args.schoolbook_no_numpy else RingLWE_Schoolbook
        rlwe_s = SchoolbookClass(n=n, q=q_n, sigma=sigma)

        # NTT backend
        rlwe_n = None
        ntt_init_err = ""
        try:
            rlwe_n = RingLWE_NTT(n=n, q=q_n, sigma=sigma)
        except Exception as e:
            ntt_init_err = f"{type(e).__name__}: {e}"

        ok_s, err_s = _supports_n(rlwe_s)
        if rlwe_n is None:
            ok_n, err_n = False, ntt_init_err
        else:
            ok_n, err_n = _supports_n(rlwe_n)

        # ring consistency
        if ok_s and ok_n:
            cons = _bench_ring_consistency(n=n, q=q_n, trials=args.consistency_trials, seed=seed + 999)
        else:
            cons = {"supported": False, "reason": f"skip: school_ok={ok_s} ({err_s}) | ntt_ok={ok_n} ({err_n})"}

        for method_name, ok, err, rlwe in [
            ("Schoolbook", ok_s, err_s, rlwe_s),
            ("NTT", ok_n, err_n, rlwe_n),
        ]:
            if not ok:
                print(
                    f"{n:<6}{method_name:<12}"
                    f"{'SKIP':>26}"
                    f"{'SKIP':>30}"
                    f"{'':>15}"
                    f"{'':>12}"
                    f"{'':>10}"
                    f"{'':>10}"
                )
                print(f"  -> {method_name} unsupported for n={n}: {err}")
                continue

            mul = _bench_mul_only(
                rlwe,
                trials=args.mul_trials,
                repeat=args.repeat,
                warmup=args.warmup,
                seed=seed + (0 if method_name == "Schoolbook" else 10_000) + n,
            )
            ed = _bench_end_to_end(
                rlwe,
                trials=args.ed_trials,
                repeat=args.repeat,
                warmup=args.warmup,
                seed=seed + (1 if method_name == "Schoolbook" else 10_001) + n,
            )

            peak_mem = max(mul["mul_peak_kib"], ed["ed_peak_kib"])

            print(
                f"{n:<6}{method_name:<12}"
                f"{_fmt_iqr(mul['mul_us_med'], mul['mul_us_p25'], mul['mul_us_p75']):>26}"
                f"{_fmt_iqr(ed['ed_us_med'], ed['ed_us_p25'], ed['ed_us_p75']):>30}"
                f"{peak_mem:>15.2f}"
                f"{ed['block_success_rate']:>12.4f}"
                f"{ed['bit_accuracy']:>10.4f}"
                f"{ed['throughput_blocks_s']:>10.2f}"
            )

        if cons.get("supported"):
            print(
                f"  Ring consistency (n={n}): "
                f"mismatches={cons['mismatch_count']}/{cons['trials']} "
                f"(rate={cons['mismatch_rate']:.4f}), "
                f"max_abs_diff={cons['max_abs_diff']}"
            )
            if cons["examples"]:
                print(f"  Examples: {cons['examples']}")
        else:
            print(f"  Ring consistency (n={n}): {cons.get('reason')}")
        print()

    print("Notes:")
    print("- Timing uses median with [p25,p75] over repeats; warmup runs are discarded.")
    print("- Peak memory uses tracemalloc (Python allocations), reported in KiB.")
    print("- Decryption success/bit accuracy are measured over --ed-trials messages.")
    print("- Ring consistency compares Schoolbook vs NTT multiplication outputs mod q.")
    print("- If some n values are unsupported by your current implementation, they are skipped.")


if __name__ == "__main__":
    main()