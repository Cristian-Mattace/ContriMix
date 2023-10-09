"""A module that implements a simple version of the LPIPS loss."""
import logging
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tv


class AlexNet(torch.nn.Module):
    """A class that implements the perceptual loss."""

    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout()] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PerceptualLoss(nn.Module):
    """A class that implements the perceptual loss.

    We need to implement this because the ScalingLayer() of the original implementation initialize the buffer on CPUs,
    see https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py#L150 and causes error when
    evaluating tensors on GPU.

    Args:
        verbose (optional): Display messages on the progress of True. Defaults to True.
    """

    def __init__(self, verbose=True) -> None:
        super().__init__()
        if verbose:
            print("Setting up the perceptual loss")

        self.net = AlexNet(requires_grad=False, pretrained=True)
        self.chns = [64, 192, 384, 256, 256]

        self._lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self._lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self._lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self._lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self._lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self._lins = [self._lin0, self._lin1, self._lin2, self._lin3, self._lin4]
        self._lins = nn.ModuleList(self._lins)

        self.load_state_dict(torch.load(str(Path(__file__).parent / "alex.pth"), map_location="cpu"), strict=False)
        self.eval()

    def forward(self, im0: torch.Tensor, im1: torch.Tensor) -> torch.Tensor:
        im0 = 2 * im0 - 1
        im1 = 2 * im1 - 1
        outs0, outs1 = self.net.forward(im0), self.net.forward(im1)
        feats0, feats1, diffs = {}, {}, {}

        num_layers = len(self.chns)
        for layer_idx in range(num_layers):
            feats0[layer_idx], feats1[layer_idx] = self._normalize_tensor(outs0[layer_idx]), self._normalize_tensor(
                outs1[layer_idx]
            )
            diffs[layer_idx] = (feats0[layer_idx] - feats1[layer_idx]) ** 2

        # Not doing torch.tensor(0.0) to avoid inferring the dimensions.
        out_res = 0
        for layer_idx in range(num_layers):
            out_res += self._spatial_average(self._lins[layer_idx](diffs[layer_idx]), keep_dim=False)

        return out_res

    @staticmethod
    def _normalize_tensor(in_feat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
        return in_feat / (norm_factor + eps)

    @staticmethod
    def _spatial_average(in_tens: torch.Tensor, keep_dim: bool = True) -> None:
        """Spatially average over the X, Y dimension of the image."""
        return in_tens.mean([2, 3], keepdim=keep_dim)
