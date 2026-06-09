import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from qldd.data import make_code_data, verify_contract, sample
from qldd.baseline import build_matching, logical_error_rate, residual_is_logical

def test_check_matrix_contract():
    for d in (3, 5):
        rep = verify_contract(make_code_data(distance=d, p=0.02), shots=2048)
        assert rep["s_equals_He"] and rep["l_equals_Le"]

def test_mwpm_below_threshold_helps():
    # below threshold, d=5 should beat d=3
    l3 = logical_error_rate(make_code_data(distance=3, p=0.015),
                            20000, seed=1)["ler"]
    l5 = logical_error_rate(make_code_data(distance=5, p=0.015),
                            20000, seed=1)["ler"]
    assert l5 < l3

def test_residual_classifier_self_consistency():
    code = make_code_data(distance=3, p=0.03)
    e, s, l = sample(code, 1000, seed=3)
    # a perfect guess (= true error) must clear syndrome and never be logical
    clears, logical = residual_is_logical(code, e, e)
    assert clears.all() and (~logical).all()
