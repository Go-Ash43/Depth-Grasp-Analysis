"""GRConvLite: the < 1M-parameter detector for R / A0 / M / Astar (SPEC_trajectory Module 1 item 3;
step 5 proposal rev 2 section 4, frozen in configs/train_v1.json "model").

Layout follows GR-ConvNet's GenerativeResnet (Kumra et al.; local reference
~/VComp/GRConv/robotic-grasping/inference/models/grconvnet3.py and grasp_model.py):

  conv1  Conv2d(in, c, 9, s1, p4)        -> BN -> ReLU      224
  conv2  Conv2d(c, 2c, 4, s2, p1)        -> BN -> ReLU      112
  conv3  Conv2d(2c, 4c, 4, s2, p1)       -> BN -> ReLU       56
  res xN conv3x3 -> BN -> ReLU -> conv3x3 -> BN, + input, NO ReLU after the add (reference)
  up1    ConvTranspose2d(4c, 2c, 4, s2, p1, op0) -> BN -> ReLU  112
  up2    ConvTranspose2d(2c, c, 4, s2, p1, op0)  -> BN -> ReLU  224
  conv6  ConvTranspose2d(c, c, 9, s1, p4): linear, no BN (reference)
  heads  4 x Conv2d(c, 1, 3, p1): Q, cos 2t, sin 2t, width; no BN, no activation

The width head is unbounded in both directions: width = opening / W_NORM may exceed 1 (configs/labels.json),
and a non-positive width decodes as invalid (data/decode.py), never clamped.
Init: xavier_uniform_(gain 1) on every Conv2d / ConvTranspose2d weight, PyTorch-default biases and BN
(as the reference). build_model seeds torch immediately before construction.

DEVIATIONS from the reference (flagged): channels 24/48/96 (not 32/64/128) and 4 residual blocks (not 5)
to stay under 1M parameters; 3x3 heads with padding 1 (not 2x2 trimming 225 -> 224); up2 padding 1 and
no output_padding anywhere (exact 56 -> 112 -> 224); no dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import MODELS

HEADS = ("q", "cos2", "sin2", "width")
CHANNELS_PER_PLANE = {"rgb": 3, "depth": 1, "flag": 1, "dstar": 1}


def in_channels(name):
    """Input channels of a named model, from the dataset's whitelist (D4)."""
    if name not in MODELS:
        raise ValueError(f"model must be one of {tuple(MODELS)}, got {name!r}")
    return sum(CHANNELS_PER_PLANE[c] for c in MODELS[name])


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = self.bn2(self.conv2(x))
        return x + x_in                               # no ReLU after the addition (reference)


class GRConvLite(nn.Module):
    def __init__(self, in_ch, c=24, n_res=4):
        super().__init__()
        self.in_ch = int(in_ch)
        self.conv1 = nn.Conv2d(in_ch, c, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, 2 * c, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * c)
        self.conv3 = nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * c)
        self.res = nn.Sequential(*[ResidualBlock(4 * c) for _ in range(n_res)])
        self.up1 = nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1, output_padding=0)
        self.bn4 = nn.BatchNorm2d(2 * c)
        self.up2 = nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1, output_padding=0)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.ConvTranspose2d(c, c, 9, stride=1, padding=4)
        self.heads = nn.ModuleDict({h: nn.Conv2d(c, 1, 3, padding=1) for h in HEADS})
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        if x_in.dim() != 4 or x_in.shape[1] != self.in_ch:
            raise ValueError(f"expected [B, {self.in_ch}, H, W], got {tuple(x_in.shape)}")
        h, w = x_in.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError(f"H and W must be multiples of 4, got {h} x {w}")
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res(x)
        x = F.relu(self.bn4(self.up1(x)))
        x = F.relu(self.bn5(self.up2(x)))
        x = self.conv6(x)
        out = {k: head(x) for k, head in self.heads.items()}
        if out["q"].shape[-2:] != (h, w):
            raise AssertionError(f"output {tuple(out['q'].shape[-2:])} != input {(h, w)}")
        return out


def build_model(name, seed, c=24, n_res=4):
    """Seeded construction: torch.manual_seed(seed) immediately before the model is built."""
    torch.manual_seed(int(seed))
    return GRConvLite(in_channels(name), c=c, n_res=n_res)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
