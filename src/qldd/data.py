"""Surface-code data pipeline: rotated surface code, phenomenological-style
noise, single-sector Z-memory. Works in DEM fault-mechanism space.

Contract: e (N, n_err), s (N, n_det), l (N, n_obs), all uint8, with
s = H e (mod 2) and l = L e (mod 2); priors gives Pr[e_j = 1].

Noise: depolarizing on data qubits; with Z-type detectors only, this reduces to
the single-sector bit-flip matching problem with effective flip prob ~2p/3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import stim


@dataclass
class CodeData:
    """A fully-specified decoding problem instance + cached check matrices."""

    distance: int
    rounds: int
    p: float                      # data-error strength (depolarizing arg)
    q: float                      # measurement-flip probability
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel
    H: np.ndarray                 # (n_det, n_err) uint8
    L: np.ndarray                 # (n_obs, n_err) uint8
    priors: np.ndarray            # (n_err,) float64
    det_coords: np.ndarray        # (n_det, 3) float  (x, y, t)
    err_coords: np.ndarray        # (n_err, 3) float  (edge midpoints)
    _sampler: object = field(default=None, repr=False)

    @property
    def n_err(self) -> int:
        return self.H.shape[1]

    @property
    def n_det(self) -> int:
        return self.H.shape[0]

    @property
    def n_obs(self) -> int:
        return self.L.shape[0]


def build_circuit(
    distance: int,
    rounds: int,
    p: float,
    q: Optional[float] = None,
    data_noise_channel: str = "depolarize",
) -> stim.Circuit:
    """Rotated-surface-code Z-memory circuit; q defaults to p. "bitflip" is
    reserved and currently routes to depolarize."""
    if q is None:
        q = p
    if distance % 2 == 0:
        raise ValueError("distance must be odd")
    if data_noise_channel not in ("depolarize", "bitflip"):
        raise ValueError(f"unknown data_noise_channel {data_noise_channel!r}")

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        before_round_data_depolarization=p,
        before_measure_flip_probability=q,
        after_reset_flip_probability=0.0,
        after_clifford_depolarization=0.0,
    )
    return circuit


def dem_to_check_matrices(dem: stim.DetectorErrorModel):
    """H (n_det x n_err), L (n_obs x n_err), priors from a flattened DEM; one
    column per "error" instruction."""
    flat = dem.flattened()
    n_det = flat.num_detectors
    n_obs = flat.num_observables

    cols_det: list[list[int]] = []
    cols_obs: list[list[int]] = []
    priors: list[float] = []

    for inst in flat:
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets: list[int] = []
        obs: list[int] = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
            # separators ignored: decomposed error = one mechanism here
        cols_det.append(dets)
        cols_obs.append(obs)
        priors.append(prob)

    n_err = len(priors)
    H = np.zeros((n_det, n_err), dtype=np.uint8)
    L = np.zeros((n_obs, n_err), dtype=np.uint8)
    for j, (dets, obs) in enumerate(zip(cols_det, cols_obs)):
        for d in dets:
            H[d, j] ^= 1
        for o in obs:
            L[o, j] ^= 1
    return H, L, np.asarray(priors, dtype=np.float64)


def _error_coords(dem: stim.DetectorErrorModel, det_coords: np.ndarray,
                  H: np.ndarray) -> np.ndarray:
    """Position per fault mechanism = midpoint of flipped detectors; boundary
    errors nudged outward in x; no-detector errors get NaN."""
    n_err = H.shape[1]
    coords = np.full((n_err, 3), np.nan, dtype=np.float64)
    for j in range(n_err):
        dets = np.nonzero(H[:, j])[0]
        if len(dets) == 0:
            continue
        coords[j] = det_coords[dets].mean(axis=0)
        if len(dets) == 1:
            coords[j, 0] -= 0.5  # boundary edge: offset toward the boundary
    return coords


def make_code_data(
    distance: int,
    rounds: Optional[int] = None,
    p: float = 0.01,
    q: Optional[float] = None,
    data_noise_channel: str = "depolarize",
) -> CodeData:
    if rounds is None:
        rounds = distance
    circuit = build_circuit(distance, rounds, p, q, data_noise_channel)
    dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True)
    H, L, priors = dem_to_check_matrices(dem)

    coord_dict = circuit.get_detector_coordinates()
    n_det = dem.num_detectors
    det_coords = np.zeros((n_det, 3), dtype=np.float64)
    for d in range(n_det):
        c = coord_dict.get(d, [0.0, 0.0, 0.0])
        # pad/truncate to (x, y, t)
        c = (list(c) + [0.0, 0.0, 0.0])[:3]
        det_coords[d] = c
    err_coords = _error_coords(dem, det_coords, H)

    return CodeData(
        distance=distance, rounds=rounds, p=p, q=(p if q is None else q),
        circuit=circuit, dem=dem, H=H, L=L, priors=priors,
        det_coords=det_coords, err_coords=err_coords,
        _sampler=dem.compile_sampler(),
    )


def sample(code: CodeData, shots: int, seed: Optional[int] = None):
    """Returns (e, s, l) uint8 arrays of shape (shots, n_err/n_det/n_obs)."""
    sampler = code._sampler if seed is None else code.dem.compile_sampler(seed=seed)
    det, obs, errs = sampler.sample(shots=shots, return_errors=True)
    e = errs.astype(np.uint8)
    s = det.astype(np.uint8)
    l = obs.astype(np.uint8)
    return e, s, l


def verify_contract(code: CodeData, shots: int = 4096, seed: int = 0) -> dict:
    """Assert s = H e and l = L e (mod 2) on random samples. Returns a report."""
    e, s, l = sample(code, shots=shots, seed=seed)
    s_pred = (code.H @ e.T) % 2          # (n_det, shots)
    l_pred = (code.L @ e.T) % 2          # (n_obs, shots)
    s_ok = np.array_equal(s_pred.T.astype(np.uint8), s)
    l_ok = np.array_equal(l_pred.T.astype(np.uint8), l)
    report = {
        "distance": code.distance, "rounds": code.rounds,
        "n_err": code.n_err, "n_det": code.n_det, "n_obs": code.n_obs,
        "syndrome_density": float(s.mean()),
        "error_density": float(e.mean()),
        "logical_flip_rate_raw": float(l.mean()),
        "s_equals_He": bool(s_ok),
        "l_equals_Le": bool(l_ok),
    }
    return report


if __name__ == "__main__":
    for d in (3, 5, 7):
        code = make_code_data(distance=d, p=0.01)
        rep = verify_contract(code)
        print(rep)
