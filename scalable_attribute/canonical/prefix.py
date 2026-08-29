import sys
from contextlib import contextmanager
from dataclasses import dataclass

import MinkowskiEngine as ME
import torch


@contextmanager
def _unicorn_args(scale, stage, vmode):
    argv = sys.argv
    sys.argv = [
        argv[0], "--scale", str(scale), "--stage", str(stage),
        "--Vmode", str(vmode),
    ]
    try:
        yield
    finally:
        sys.argv = argv


@dataclass
class PrefixState:
    x4: ME.SparseTensor
    f4: ME.SparseTensor
    d4: ME.SparseTensor
    x5p: ME.SparseTensor
    f5p: ME.SparseTensor
    d5p: ME.SparseTensor


class FrozenUnicornPrefix(torch.nn.Module):
    """Released Unicorn through r4 plus its decoder-known native transition."""

    residual_stages = 4

    def __init__(self, checkpoint, scale=5, stage=1, vmode=1):
        super().__init__()
        if scale != 5 or stage != 1:
            raise ValueError("Canonical r4 split requires scale=5 and stage=1")
        with _unicorn_args(scale, stage, vmode):
            from lossy_attribute.model import MultiscaleVAE

        self.model = MultiscaleVAE(stage=stage)
        state = torch.load(checkpoint, map_location="cpu")["model"]
        expected = self.model.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise RuntimeError(
                "Released checkpoint mismatch: missing={} unexpected={}".format(
                    missing, unexpected))
        self.model.load_state_dict(state)
        self.model.requires_grad_(False)
        self.model.eval()
        self.eval()

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, attribute, lmb):
        self.model.eval()
        output = self.model(
            attribute, training=False, lmb=lmb, real_coding=False,
            max_residual_stages=self.residual_stages, return_state=True)
        state = output["state"]
        return self._complete_state(state["x"], state["f"], state["dec"])

    @torch.no_grad()
    def hard_forward(self, attribute, lmb, return_details=False):
        self.model.eval()
        encoded, x_low, gpcc_bits = self.model(
            attribute, training=False, lmb=lmb, encode=True,
            max_residual_stages=self.residual_stages)
        x0 = ME.SparseTensor(
            features=torch.zeros_like(attribute.F),
            coordinate_map_key=attribute.coordinate_map_key,
            coordinate_manager=attribute.coordinate_manager,
            device=attribute.device,
        )
        x4, f4, d4 = self.model.decode(
            x0=x0, x_low=x_low, enc_set_list=encoded, lmb=lmb,
            max_residual_stages=self.residual_stages, return_state=True)
        stream_bits = [int(len(item["strings"]) * 8) for item in encoded]
        details = {
            "bits_xlow": int(gpcc_bits),
            "residual_bits": stream_bits,
            "num_residual_streams": len(encoded),
            "base_bits": int(gpcc_bits + sum(stream_bits)),
        }
        state = self._complete_state(x4, f4, d4)
        return (state, details) if return_details else (state, details["base_bits"])

    @torch.no_grad()
    def _complete_state(self, x4, f4, d4):
        x5p, f5p, d5p = self.model.prepare_next_scale(
            x4, f4, d4, self.residual_stages)
        return PrefixState(x4=x4, f4=f4, d4=d4,
                           x5p=x5p, f5p=f5p, d5p=d5p)

    def synthesize(self, f5p, compensation):
        feature = self.model.VAE.fuseNet(f5p + compensation)
        correction = self.model.VAE.outNet(feature)
        return feature, correction
