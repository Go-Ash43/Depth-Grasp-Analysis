"""models/grconv_lite.py: frozen architecture (step 5 proposal rev 2 section 4)."""
import math

import pytest
import torch

from models.grconv_lite import HEADS, GRConvLite, ResidualBlock, build_model, in_channels, n_params

COUNTS = {"R": 904_276, "A0": 906_220, "M": 908_164, "Astar": 906_220}


@pytest.mark.parametrize("name", sorted(COUNTS))
def test_exact_parameter_counts_under_1m(name):
    m = build_model(name, 0)
    assert n_params(m) == COUNTS[name] < 1_000_000
    assert m.in_ch == in_channels(name)


@pytest.mark.parametrize("b", [1, 2])
def test_224_in_224_out_all_heads(b):
    m = build_model("M", 0).eval()
    with torch.no_grad():
        out = m(torch.randn(b, 5, 224, 224))
    assert set(out) == set(HEADS)
    for v in out.values():
        assert tuple(v.shape) == (b, 1, 224, 224)


def test_wrong_channels_raise():
    with pytest.raises(ValueError):
        build_model("A0", 0)(torch.zeros(1, 5, 224, 224))


@pytest.mark.parametrize("bias", [5.0, -3.0])
def test_heads_are_linear_and_width_unbounded(bias):
    m = build_model("A0", 0).eval()
    with torch.no_grad():
        for h in HEADS:
            m.heads[h].weight.zero_()
            m.heads[h].bias.fill_(bias)
        out = m(torch.randn(1, 4, 224, 224))
    for h in HEADS:
        assert torch.all(out[h] == bias)          # no sigmoid / clamp / softplus on any head


def test_residual_has_no_relu_after_add():
    blk = ResidualBlock(8).eval()
    with torch.no_grad():
        blk.conv2.weight.zero_()
        blk.conv2.bias.zero_()
        x = -torch.rand(1, 8, 6, 6) - 0.5          # all negative
        assert torch.equal(blk(x), x)               # a ReLU after the add would zero these


def test_heads_and_conv6_have_no_bn():
    m = build_model("M", 0)
    assert isinstance(m.conv6, torch.nn.ConvTranspose2d)
    names = [n for n, _ in m.named_modules()]
    assert not any(n.startswith("heads.") and "bn" in n for n in names)


def test_xavier_init_not_framework_default():
    m = build_model("M", 0)
    w = m.res[0].conv1.weight.detach()            # 96 x 96 x 3 x 3
    fan_in = fan_out = 96 * 9
    xavier_std = math.sqrt(2.0 / (fan_in + fan_out))          # 0.034
    default_std = (1.0 / math.sqrt(fan_in)) / math.sqrt(3.0)  # kaiming_uniform(a=sqrt 5): 0.0196
    s = float(w.std())
    assert abs(s - xavier_std) < 0.1 * xavier_std and abs(s - default_std) > 0.3 * default_std
    assert float(w.abs().max()) <= math.sqrt(6.0 / (fan_in + fan_out)) + 1e-7


def test_seeded_init_is_reproducible():
    a, b, c = build_model("A0", 3), build_model("A0", 3), build_model("A0", 4)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not all(torch.equal(sa[k], sc[k]) for k in sa)


def test_odd_size_refused():
    with pytest.raises(ValueError):
        GRConvLite(3)(torch.zeros(1, 3, 222, 222))
