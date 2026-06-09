"""MWPM baseline (PyMatching) and LER evaluation. PyMatching decodes from the
same Stim DEM as the data pipeline, so comparisons share noise realizations.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pymatching

from .data import CodeData, make_code_data, sample


def build_matching(code: CodeData) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(code.dem)


def logical_error_rate(
    code: CodeData,
    shots: int,
    matching: Optional[pymatching.Matching] = None,
    seed: int = 0,
) -> dict:
    """MWPM decode; returns per-experiment LER (fraction of shots where the
    predicted observable flip != true flip) and stderr."""
    if matching is None:
        matching = build_matching(code)
    e, s, l = sample(code, shots=shots, seed=seed)
    pred = matching.decode_batch(s)                 # (shots, n_obs) uint8
    pred = np.asarray(pred, dtype=np.uint8)
    fail = np.any(pred != l, axis=1)                # logical failure per shot
    n_fail = int(fail.sum())
    ler = n_fail / shots
    stderr = float(np.sqrt(max(ler * (1 - ler), 1e-12) / shots))
    return {"ler": ler, "stderr": stderr, "n_fail": n_fail, "shots": shots}


def residual_is_logical(code: CodeData, e_true: np.ndarray, e_guess: np.ndarray):
    """Classify the residual E_r = e_true ^ e_guess. Returns boolean (N,) arrays
    (clears_syndrome: H E_r = 0, logical_failure: L E_r != 0)."""
    er = (e_true.astype(np.uint8) ^ e_guess.astype(np.uint8))
    h_res = (code.H @ er.T) % 2                      # (n_det, N)
    l_res = (code.L @ er.T) % 2                      # (n_obs, N)
    clears = ~np.any(h_res, axis=0)
    logical = np.any(l_res, axis=0)
    return clears, logical


def threshold_sweep(
    distances,
    ps,
    shots: int,
    rounds: Optional[int] = None,
    q: Optional[float] = None,
    seed: int = 0,
) -> dict:
    """MWPM LER over a (distance, p) grid; curves cross near the threshold p*."""
    results = {}
    for d in distances:
        row = []
        for p in ps:
            code = make_code_data(distance=d, rounds=rounds, p=p, q=q)
            m = build_matching(code)
            r = logical_error_rate(code, shots=shots, matching=m, seed=seed)
            row.append(r["ler"])
        results[d] = np.asarray(row)
    return {"distances": list(distances), "ps": np.asarray(ps), "ler": results}


if __name__ == "__main__":
    ps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    sweep = threshold_sweep(distances=[3, 5, 7], ps=ps, shots=20000)
    print("p:        " + "  ".join(f"{p:6.3f}" for p in sweep["ps"]))
    for d in sweep["distances"]:
        print(f"d={d}:  " + "  ".join(f"{x:6.4f}" for x in sweep["ler"][d]))
