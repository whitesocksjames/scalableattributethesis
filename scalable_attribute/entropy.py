from dataclasses import dataclass

import torch
import MinkowskiEngine as ME

from basic_models.conditional_entropy_model import SymmetricConditional
from data_utils.sparse_tensor import sort_sparse_tensor


@dataclass
class EnhancementEncoded:
    strings: bytes
    min_v: object
    max_v: object


class EnhancementEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditional = SymmetricConditional()

    @staticmethod
    def _aligned(source, target):
        if len(source) != len(target):
            raise RuntimeError("EL latent and conditional prior supports differ")
        source = sort_sparse_tensor(source, target=target)
        if not torch.equal(source.C, target.C):
            raise RuntimeError("EL latent and conditional prior coordinates differ")
        return source

    def forward(self, y_E, mu, sigma):
        mu = self._aligned(mu, y_E)
        sigma = self._aligned(sigma, y_E)
        mode = "noise" if self.training else "symbols"
        y_hat_F, likelihood = self.conditional(
            y_E.F, mu.F, sigma.F.abs().clamp(min=1e-8), quantize_mode=mode)
        y_hat = ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=y_E.coordinate_map_key,
            coordinate_manager=y_E.coordinate_manager,
            device=y_E.device,
        )
        return y_hat, likelihood

    @torch.no_grad()
    def encode(self, y_E, mu, sigma):
        mu = self._aligned(mu, y_E)
        sigma = self._aligned(sigma, y_E)
        strings, min_v, max_v = self.conditional.compress(
            y_E.F, mu.F, sigma.F.abs().clamp(min=1e-8))
        y_hat_F = self.conditional._quantize(y_E.F, mode="symbols")
        y_hat = ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=y_E.coordinate_map_key,
            coordinate_manager=y_E.coordinate_manager,
            device=y_E.device,
        )
        return y_hat, EnhancementEncoded(strings, min_v, max_v)

    @torch.no_grad()
    def decode(self, encoded, mu, sigma):
        if not torch.equal(mu.C, sigma.C):
            sigma = self._aligned(sigma, mu)
        self.conditional._channels = mu.F.shape[1]
        y_hat_F = self.conditional.decompress(
            encoded.strings,
            mu.F,
            sigma.F.abs().clamp(min=1e-8),
            encoded.min_v,
            encoded.max_v,
            channels=mu.F.shape[1],
        )
        return ME.SparseTensor(
            features=y_hat_F,
            coordinate_map_key=mu.coordinate_map_key,
            coordinate_manager=mu.coordinate_manager,
            device=mu.device,
        )
