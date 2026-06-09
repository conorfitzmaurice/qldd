"""Sanity check: data-contract verification + MWPM threshold crossing. CPU ok."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qldd.data import make_code_data, verify_contract
from qldd.baseline import threshold_sweep


def main():
    print("== data contract (s = He, l = Le) ==")
    ok = True
    for d in (3, 5, 7):
        rep = verify_contract(make_code_data(distance=d, p=0.02))
        ok &= rep["s_equals_He"] and rep["l_equals_Le"]
        print(f"  d={d}: s=He {rep['s_equals_He']}  l=Le {rep['l_equals_Le']}  "
              f"n_err={rep['n_err']} n_det={rep['n_det']}")
    assert ok, "DATA CONTRACT VIOLATED"

    print("\n== MWPM threshold sweep (expect a crossing near p* ~ 0.04) ==")
    ps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    sw = threshold_sweep(distances=[3, 5, 7], ps=ps, shots=20000)
    print("  p:      " + "  ".join(f"{p:6.3f}" for p in sw["ps"]))
    for d in sw["distances"]:
        print(f"  d={d}:  " + "  ".join(f"{x:6.4f}" for x in sw["ler"][d]))
    # crossing check: ordering flips between low-p and high-p
    lo = {d: sw["ler"][d][0] for d in sw["distances"]}
    hi = {d: sw["ler"][d][-1] for d in sw["distances"]}
    cross = (lo[7] < lo[3]) and (hi[7] > hi[3])
    print(f"\n  threshold crossing present: {cross}")
    assert cross, "NO THRESHOLD CROSSING -- pipeline suspect"
    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
