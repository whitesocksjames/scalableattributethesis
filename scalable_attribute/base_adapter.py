import sys
from contextlib import contextmanager

import torch
import MinkowskiEngine as ME


@contextmanager
def _unicorn_args(scale, stage, vmode):
    argv = sys.argv
    sys.argv = [argv[0], "--scale", str(scale), "--stage", str(stage), "--Vmode", str(vmode)]
    try:
        yield
    finally:
        sys.argv = argv


class BaseAdapter(torch.nn.Module):
    """Small boundary around the released Unicorn-v1 Attribute model."""

    def __init__(self, checkpoint, scale=5, stage=1, vmode=1,
                 attribute_channels=3, feature_channels=128):
        super().__init__()
        with _unicorn_args(scale, stage, vmode):
            from lossy_attribute.model import MultiscaleVAE

        self.base = MultiscaleVAE(stage=stage)
        state = torch.load(checkpoint, map_location="cpu")["model"]
        expected = self.base.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise RuntimeError(
                "Base checkpoint mismatch: missing={} unexpected={}".format(missing, unexpected))
        self.base.load_state_dict(state)
        self.base.requires_grad_(False)
        self.base.eval()
        self.attribute_channels = attribute_channels
        self.feature_channels = feature_channels

    def train(self, mode=True):
        super().train(False)
        self.base.eval()
        return self

    def _check(self, A, B, F_U):
        if (A.F.shape[1] != self.attribute_channels
                or B.F.shape[1] != self.attribute_channels
                or F_U.F.shape[1] != self.feature_channels):
            raise RuntimeError("Unexpected A/B/F_U channel count")
        if list(A.tensor_stride) != [1, 1, 1]:
            raise RuntimeError("A must have tensor_stride 1")
        if not torch.equal(A.C, B.C) or not torch.equal(A.C, F_U.C):
            raise RuntimeError("A, B and F_U coordinates are not aligned")

    @torch.no_grad()
    def forward(self, A, base_lambda):
        self.base.eval()
        output = self.base(A, training=False, lmb=base_lambda, real_coding=False)
        B = output["out_list"][0]
        F_U = output["curr_f"]
        self._check(A, B, F_U)
        return B, F_U

    @torch.no_grad()
    def hard_reconstruct(self, A, base_lambda):
        self.base.eval()
        encoded, x_low, gpcc_bits = self.base(
            A, training=False, lmb=base_lambda, encode=True)
        x0 = ME.SparseTensor(
            features=torch.zeros_like(A.F),
            coordinate_map_key=A.coordinate_map_key,
            coordinate_manager=A.coordinate_manager,
            device=A.device,
        )
        B, F_U = self.base.decode(
            x0=x0,
            x_low=x_low,
            enc_set_list=encoded,
            lmb=base_lambda,
            return_feature=True,
        )
        self._check(A, B, F_U)
        bits = int(gpcc_bits + sum(len(item["strings"]) * 8 for item in encoded))
        return B, F_U, bits
