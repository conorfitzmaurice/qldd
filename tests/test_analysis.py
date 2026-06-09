import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qldd.data import make_code_data
from qldd.model import LocalDiffusionDecoder, ModelConfig
from qldd.analysis import conv_receptive_field_voxels, locality_report

def test_conv_rf_grows_with_depth():
    code = make_code_data(distance=9, p=0.02)  # large enough grid to see growth
    rf1 = conv_receptive_field_voxels(
        LocalDiffusionDecoder(ModelConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,
                              conv_channels=8,conv_layers=1), code))
    rf2 = conv_receptive_field_voxels(
        LocalDiffusionDecoder(ModelConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,
                              conv_channels=8,conv_layers=2), code))
    assert rf2["rf_voxels"]["x"] >= rf1["rf_voxels"]["x"]

def test_d3_locality_is_confounded():
    # d=3 grid is too small: conv is global even shallow -> not a valid locality test
    code = make_code_data(distance=3, p=0.02)
    m = LocalDiffusionDecoder(ModelConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,
                              conv_channels=8,conv_layers=1), code)
    rep = locality_report(m, code)
    assert rep["conv_is_global"] and not rep["locality_test_meaningful"]

def test_conv_off_gives_meaningful_locality_at_d7():
    code = make_code_data(distance=7, p=0.02)
    m = LocalDiffusionDecoder(ModelConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,
                              use_conv_stem=False, sigma_init=1.0), code)
    rep = locality_report(m, code)
    assert rep["locality_test_meaningful"]
    assert rep["total_effective_range_lattice"] is not None

def test_windowed_matches_dense_large_radius():
    import torch
    from qldd.model import LocalDiffusionDecoder, ModelConfig
    from qldd.data import sample
    code = make_code_data(distance=3, p=0.03)
    md = LocalDiffusionDecoder(ModelConfig(d_model=32,n_heads=2,n_layers=2,d_ff=64,
                               conv_channels=8,conv_layers=1, window_radius=None), code).eval()
    mw = LocalDiffusionDecoder(ModelConfig(d_model=32,n_heads=2,n_layers=2,d_ff=64,
                               conv_channels=8,conv_layers=1, window_radius=999.0), code).eval()
    mw.load_state_dict(md.state_dict(), strict=False)
    e,s,l = sample(code, 4, seed=1)
    with torch.no_grad():
        od = md(torch.as_tensor(s), torch.as_tensor(e))
        ow = mw(torch.as_tensor(s), torch.as_tensor(e))
    assert float((od-ow).abs().max()) < 1e-4

def test_windowed_K_shrinks_with_radius():
    from qldd.model import LocalDiffusionDecoder, ModelConfig
    code = make_code_data(distance=7, p=0.03)
    small = LocalDiffusionDecoder(ModelConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,
                                  use_conv_stem=False, window_radius=1.5), code)
    big = LocalDiffusionDecoder(ModelConfig(d_model=16,n_heads=2,n_layers=1,d_ff=32,
                                use_conv_stem=False, window_radius=4.0), code)
    assert small.window_K < big.window_K
