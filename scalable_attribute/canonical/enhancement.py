"""Independent native ResidualVAE used by the canonical Enhancement layer."""

import torch


class EnhancementVAE(torch.nn.Module):
    """A trainable ResidualVAE initialized from, but not shared with, Unicorn."""

    def __init__(self, released_vae):
        super().__init__()
        # ResidualVAE reads the released Attribute configuration at module import.
        # Reuse that already-configured class without re-importing the author module.
        self.vae = type(released_vae)(stride=[2, 2, 2])
        self.vae.load_state_dict(released_vae.state_dict(), strict=True)

    def forward(self, base, ground_truth, base_feature, prior_dec, embedding):
        return self.vae(
            x_in=base, x_gt=ground_truth, f_in=base_feature,
            prior_dec=prior_dec, training=True, emb=embedding,
            real_coding=False)

    @torch.no_grad()
    def deterministic(self, base, ground_truth, base_feature, prior_dec,
                      embedding):
        return self.vae(
            x_in=base, x_gt=ground_truth, f_in=base_feature,
            prior_dec=prior_dec, training=False, emb=embedding,
            real_coding=False)

    @torch.no_grad()
    def encode(self, base, ground_truth, base_feature, prior_dec, embedding):
        return self.vae.encode(
            x_in=base, x_gt=ground_truth, f_in=base_feature,
            prior_dec=prior_dec, emb=embedding)

    @torch.no_grad()
    def decode(self, payload, base, base_feature, prior_dec, embedding):
        return self.vae.decode(
            strings=payload["strings"], min_v=payload["min_v"],
            max_v=payload["max_v"], x_in=base, f_in=base_feature,
            prior_dec=prior_dec, emb=embedding)
