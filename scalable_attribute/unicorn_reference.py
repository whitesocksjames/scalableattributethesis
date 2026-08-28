import sys
from contextlib import contextmanager

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


class ReleasedUnicornAttribute(torch.nn.Module):
    """Narrow hard-code adapter used only by released-reference evaluation."""

    def __init__(self, checkpoint, scale=5, stage=1, vmode=1):
        super().__init__()
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

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def hard_reconstruct(self, attribute, lmb):
        encoded, x_low, gpcc_bits = self.model(
            attribute, training=False, lmb=lmb, encode=True)
        x0 = ME.SparseTensor(
            features=torch.zeros_like(attribute.F),
            coordinate_map_key=attribute.coordinate_map_key,
            coordinate_manager=attribute.coordinate_manager,
            device=attribute.device,
        )
        reconstruction = self.model.decode(
            x0=x0, x_low=x_low, enc_set_list=encoded, lmb=lmb)
        if not torch.equal(attribute.C, reconstruction.C):
            raise RuntimeError(
                "Released reconstruction coordinates do not match input")
        bits = int(gpcc_bits + sum(
            len(item["strings"]) * 8 for item in encoded))
        return reconstruction, bits
