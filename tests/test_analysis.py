import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import torch
from qldd.data import make_code_data, sample
from qldd.model import LocalDiffusionDecoder, ModelConfig, SoftLocalAttention
from qldd.analysis import conv_receptive_field_voxels, locality_report


def test_conv_rf_grows_with_depth():
    code = make_code_data(distance=9, p=0.02)
    rf1 = conv_receptive_field_voxels(
        LocalDiffusionDecoder(ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                              conv_channels=8, conv_layers=1), code))
    rf2 = conv_receptive_field_voxels(
        LocalDiffusionDecoder(ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                              conv_channels=8, conv_layers=2), code))
    assert rf2["rf_voxels"]["x"] >= rf1["rf_voxels"]["x"]


def test_d3_locality_is_confounded():
    code = make_code_data(distance=3, p=0.02)
    m = LocalDiffusionDecoder(ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                              conv_channels=8, conv_layers=1), code)
    rep = locality_report(m, code)
    assert rep["conv_is_global"] and not rep["locality_test_meaningful"]


def test_conv_off_gives_meaningful_locality_at_d7():
    code = make_code_data(distance=7, p=0.02)
    m = LocalDiffusionDecoder(ModelConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32,
                              use_conv_stem=False, xi_space_init=1.0), code)
    rep = locality_report(m, code)
    assert rep["locality_test_meaningful"]
    assert rep["total_effective_range_lattice"] is not None


def _geom(dist_s, dist_t=None, dt_signed=None):
    T = dist_s.shape[0]
    return {"dist_s": dist_s,
            "dist_t": dist_t if dist_t is not None else torch.zeros(T, T),
            "dt_signed": dt_signed if dt_signed is not None else torch.zeros(T, T)}


def test_sdpa_matches_manual_softmax():
    """SDPA + additive bias must equal hand-computed exp-weighted attention."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, xi_space_init=1.3, xi_time_init=0.7)
    attn = SoftLocalAttention(cfg).eval()
    T = 37
    x = torch.randn(3, T, cfg.d_model)
    ds = torch.rand(T, T) * 5.0; ds = (ds + ds.T) / 2; ds.fill_diagonal_(0)
    dt = torch.rand(T, T) * 3.0; dt = (dt + dt.T) / 2; dt.fill_diagonal_(0)
    with torch.no_grad():
        out = attn(x, _geom(ds, dt))
        B, h, dk = 3, cfg.n_heads, cfg.d_model // cfg.n_heads
        qkv = attn.qkv(x).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (dk ** 0.5)
        bias = -ds / attn.xi_space().view(1, h, 1, 1) \
               - dt / attn.xi_time().view(1, h, 1, 1)
        w = torch.softmax(scores + bias, dim=-1)
        ref = attn.out((w @ v).transpose(1, 2).reshape(B, T, h * dk))
    assert float((out - ref).abs().max()) < 1e-5


def test_large_xi_recovers_plain_attention():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, xi_space_init=1e6, xi_time_init=1e6)
    attn = SoftLocalAttention(cfg).eval()
    T = 25
    x = torch.randn(2, T, cfg.d_model)
    with torch.no_grad():
        out = attn(x, _geom(torch.rand(T, T) * 3.0))
        ref = attn(x, _geom(torch.zeros(T, T)))   # zero distance == no bias
    assert float((out - ref).abs().max()) < 1e-3


def test_hard_window_blocks_far_tokens():
    """With window_radius set, attention to tokens beyond R must be exactly 0:
    moving a far token must not change the output for tokens it can't reach."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, xi_space_init=2.0, window_radius=1.0)
    attn = SoftLocalAttention(cfg).eval()
    T = 10
    x = torch.randn(1, T, cfg.d_model)
    dist = torch.full((T, T), 5.0)
    dist.fill_diagonal_(0)
    dist[0, 1] = dist[1, 0] = 0.5          # only tokens 0,1 are mutually in range
    with torch.no_grad():
        out1 = attn(x, _geom(dist))
        x2 = x.clone()
        x2[0, 5] += 100.0                   # perturb a far token
        out2 = attn(x2, _geom(dist))
    assert float((out1[0, 0] - out2[0, 0]).abs().max()) < 1e-5
    assert float((out1[0, 1] - out2[0, 1]).abs().max()) < 1e-5


def test_causal_time_blocks_future():
    """With causal_time, perturbing a future-time token must not change any
    output at earlier times."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, causal_time=True)
    attn = SoftLocalAttention(cfg).eval()
    T = 12
    times = torch.arange(T).float()
    dt_signed = times[None, :] - times[:, None]      # t_key - t_query
    g = {"dist_s": torch.zeros(T, T), "dist_t": dt_signed.abs(),
         "dt_signed": dt_signed}
    x = torch.randn(1, T, cfg.d_model)
    with torch.no_grad():
        out1 = attn(x, g)
        x2 = x.clone(); x2[0, -1] += 100.0           # perturb the LAST time
        out2 = attn(x2, g)
    assert float((out1[0, :-1] - out2[0, :-1]).abs().max()) < 1e-5
    # and the last token itself (which sees everything past) does change
    assert float((out1[0, -1] - out2[0, -1]).abs().max()) > 1e-3


def test_both_xi_receive_gradients():
    code = make_code_data(distance=3, p=0.03)
    m = LocalDiffusionDecoder(ModelConfig(use_conv_stem=False), code)
    m.train()
    e, s, l = sample(code, 8, seed=1)
    out = m(torch.as_tensor(s), torch.as_tensor(e))
    out.sum().backward()
    gs = m.blocks[0].attn.raw_xi_s.grad
    gt = m.blocks[0].attn.raw_xi_t.grad
    assert gs is not None and torch.isfinite(gs).all() and gs.abs().sum() > 0
    assert gt is not None and torch.isfinite(gt).all() and gt.abs().sum() > 0
