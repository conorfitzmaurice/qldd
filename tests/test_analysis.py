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
                              use_conv_stem=False, xi_init=1.0), code)
    rep = locality_report(m, code)
    assert rep["locality_test_meaningful"]
    assert rep["total_effective_range_lattice"] is not None


def test_sdpa_matches_manual_softmax():
    """SDPA + additive bias must equal hand-computed exp(-d/xi)-weighted attn."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, xi_init=1.3)
    attn = SoftLocalAttention(cfg).eval()
    T = 37
    x = torch.randn(3, T, cfg.d_model)
    dist = torch.rand(T, T) * 5.0
    dist = (dist + dist.T) / 2
    dist.fill_diagonal_(0)
    with torch.no_grad():
        out = attn(x, dist)
        # manual computation
        B, h, dk = 3, cfg.n_heads, cfg.d_model // cfg.n_heads
        qkv = attn.qkv(x).view(B, T, 3, h, dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (dk ** 0.5)
        xi = attn.xi().view(1, h, 1, 1)
        w = torch.softmax(scores + (-dist / xi), dim=-1)   # == exp(-d/xi) scaling
        ref = attn.out((w @ v).transpose(1, 2).reshape(B, T, h * dk))
    assert float((out - ref).abs().max()) < 1e-5


def test_large_xi_recovers_plain_attention():
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, xi_init=1e6)
    attn = SoftLocalAttention(cfg).eval()
    T = 25
    x = torch.randn(2, T, cfg.d_model)
    dist = torch.rand(T, T) * 3.0
    with torch.no_grad():
        out = attn(x, dist)
        ref = attn(x, torch.zeros(T, T))   # zero distance == no bias
    assert float((out - ref).abs().max()) < 1e-3


def test_hard_window_blocks_far_tokens():
    """With window_radius set, attention to tokens beyond R must be exactly 0:
    moving a far token must not change the output for tokens it can't reach."""
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                      use_conv_stem=False, xi_init=2.0, window_radius=1.0)
    attn = SoftLocalAttention(cfg).eval()
    T = 10
    x = torch.randn(1, T, cfg.d_model)
    dist = torch.full((T, T), 5.0)
    dist.fill_diagonal_(0)
    dist[0, 1] = dist[1, 0] = 0.5          # only tokens 0,1 are mutually in range
    with torch.no_grad():
        out1 = attn(x, dist)
        x2 = x.clone()
        x2[0, 5] += 100.0                   # perturb a far token
        out2 = attn(x2, dist)
    assert float((out1[0, 0] - out2[0, 0]).abs().max()) < 1e-5
    assert float((out1[0, 1] - out2[0, 1]).abs().max()) < 1e-5


def test_xi_receives_gradients():
    code = make_code_data(distance=3, p=0.03)
    m = LocalDiffusionDecoder(ModelConfig(use_conv_stem=False), code)
    e, s, l = sample(code, 8, seed=1)
    out = m(torch.as_tensor(s), torch.as_tensor(e))
    out.sum().backward()
    g = m.blocks[0].attn.raw_xi.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
