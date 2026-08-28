import torch
import MinkowskiEngine as ME

from basic_models.backbone import Backbone
from data_utils.sparse_tensor import sort_sparse_tensor
from scalable_attribute.entropy import EnhancementEntropy


class ConditionalEnhancementLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        common = dict(
            channels=c.hidden_channels,
            block_type=c.block_type,
            block_layers=c.block_layers,
            kernel_size=c.kernel_size,
        )
        self.residual_stem = Backbone(
            scale=0, in_channels=c.attribute_channels,
            out_channels=c.hidden_channels, **common)
        fusion_channels = c.hidden_channels + c.attribute_channels + c.base_feature_channels
        self.analysis_fusion = Backbone(
            scale=0, in_channels=fusion_channels,
            out_channels=c.hidden_channels, **common)
        self.analysis_transform = Backbone(
            scale=c.analysis_scale, in_channels=c.hidden_channels,
            out_channels=c.latent_channels, **common)

        condition_channels = c.attribute_channels + c.base_feature_channels
        self.prior_transform = Backbone(
            scale=c.analysis_scale, in_channels=condition_channels,
            out_channels=c.hidden_channels, **common)
        self.mu_head = ME.MinkowskiLinear(c.hidden_channels, c.latent_channels)
        self.sigma_head = ME.MinkowskiLinear(c.hidden_channels, c.latent_channels)
        self.entropy = EnhancementEntropy()

        self.synthesis_transform = Backbone(
            scale=-c.analysis_scale, in_channels=c.latent_channels,
            out_channels=c.hidden_channels, **common)
        synthesis_channels = {
            "b_fu": fusion_channels,
            "b": c.hidden_channels + c.attribute_channels,
            "none": c.hidden_channels,
        }[c.synthesis_condition]
        self.synthesis_condition = c.synthesis_condition
        self.zero_centered_synthesis = c.zero_centered_synthesis
        self.synthesis_fusion = Backbone(
            scale=0, in_channels=synthesis_channels,
            out_channels=c.hidden_channels, **common)
        self.out_net = Backbone(
            scale=0,
            in_channels=c.hidden_channels,
            channels=c.hidden_channels,
            out_channels=c.attribute_channels,
            block_type="linear",
            block_layers=c.block_layers,
            kernel_size=c.kernel_size,
        )

    @staticmethod
    def _align(source, target, name):
        if len(source) != len(target):
            raise RuntimeError(name + " coordinate support size mismatch")
        source = sort_sparse_tensor(source, target=target)
        if not torch.equal(source.C, target.C):
            raise RuntimeError(name + " coordinate support mismatch")
        return source

    def _analysis(self, A, B, F_U):
        E = A - B
        residual_feature = self.residual_stem(E)
        fused = self.analysis_fusion(ME.cat([residual_feature, B, F_U]))
        return self.analysis_transform(fused)

    def _prior(self, B, F_U, target=None):
        prior_feature = self.prior_transform(ME.cat([B, F_U]))
        mu = self.mu_head(prior_feature)
        sigma = self.sigma_head(prior_feature)
        if target is not None:
            mu = self._align(mu, target, "mu")
            sigma = self._align(sigma, target, "sigma")
        return mu, sigma

    def _decode_delta(self, y_hat, B, F_U):
        decoded = self.synthesis_transform(y_hat)
        decoded = self._align(decoded, B, "EL decoder")
        if self.synthesis_condition == "b_fu":
            synthesis_input = ME.cat([decoded, B, F_U])
        elif self.synthesis_condition == "b":
            synthesis_input = ME.cat([decoded, B])
        else:
            synthesis_input = decoded
        fused = self.synthesis_fusion(synthesis_input)
        return self.out_net(fused)

    def _synthesis(self, y_hat, B, F_U):
        delta_A = self._decode_delta(y_hat, B, F_U)
        if self.zero_centered_synthesis:
            zero_y_hat = ME.SparseTensor(
                features=torch.zeros_like(y_hat.F),
                coordinate_map_key=y_hat.coordinate_map_key,
                coordinate_manager=y_hat.coordinate_manager,
                device=y_hat.device,
            )
            delta_A = delta_A - self._decode_delta(zero_y_hat, B, F_U)
        Full = B + delta_A
        return delta_A, Full

    def forward(self, A, B, F_U):
        y_E = self._analysis(A, B, F_U)
        mu, sigma = self._prior(B, F_U, target=y_E)
        y_hat, likelihood = self.entropy(y_E, mu, sigma)
        delta_A, Full = self._synthesis(y_hat, B, F_U)
        return {
            "y_E": y_E,
            "y_hat": y_hat,
            "delta_A": delta_A,
            "Full": Full,
            "likelihood": likelihood,
        }

    @torch.no_grad()
    def hard_encode(self, A, B, F_U):
        y_E = self._analysis(A, B, F_U)
        mu, sigma = self._prior(B, F_U, target=y_E)
        y_hat, encoded = self.entropy.encode(y_E, mu, sigma)
        _, Full = self._synthesis(y_hat, B, F_U)
        return encoded, Full

    @torch.no_grad()
    def hard_decode(self, encoded, B, F_U):
        mu, sigma = self._prior(B, F_U)
        y_hat = self.entropy.decode(encoded, mu, sigma)
        return self._synthesis(y_hat, B, F_U)[1]
