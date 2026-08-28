import torch

from scalable_attribute.base_adapter import BaseAdapter


class C2NativeEnhancement(torch.nn.Module):
    """One appended Unicorn-native ResidualVAE; Base remains untouched."""

    def __init__(self, released_vae):
        super().__init__()
        self.vae = released_vae.__class__(stride=[2, 2, 2])
        self.vae.load_state_dict(released_vae.state_dict())

    def forward(self, A, B, F_U, D_U, emb):
        return self.vae(
            x_in=B, x_gt=A, f_in=F_U, prior_dec=D_U,
            training=self.training, emb=emb, real_coding=False)

    def raw_symbols(self, A, B, emb):
        latent = self.vae.encoder(A - B)
        latent = self.vae.EQlayer(latent, emb)
        return torch.round(latent.F)


class C2ScalableAttributeModel(torch.nn.Module):
    def __init__(self, base_checkpoint, base_scale=5, base_stage=1, base_vmode=1):
        super().__init__()
        self.base_adapter = BaseAdapter(
            base_checkpoint, scale=base_scale, stage=base_stage, vmode=base_vmode)
        self.enhancement = C2NativeEnhancement(self.base_adapter.base.VAE)

    def forward(self, A, base_lambda):
        B, F_U, D_U = self.base_adapter.forward_state(A, base_lambda)
        emb = self.base_adapter.embedding(base_lambda, A.device)
        output = self.enhancement(A, B, F_U, D_U, emb)
        output.update({"B": B, "F_U": F_U, "D_U": D_U, "Full": output["x_out"]})
        return output


class C2ScalableAttributeCoder:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def test(self, A, base_lambda):
        self.model.eval()
        B, F_U, D_U, base_bits = self.model.base_adapter.hard_reconstruct_state(
            A, base_lambda)
        emb = self.model.base_adapter.embedding(base_lambda, A.device)
        symbols = self.model.enhancement.raw_symbols(A, B, emb)
        encoded = self.model.enhancement.vae.encode(B, A, F_U, D_U, emb)
        decoded = self.model.enhancement.vae.decode(
            encoded["strings"], encoded["min_v"], encoded["max_v"],
            B, F_U, D_U, emb)
        max_difference = float(
            (encoded["x_out"].F - decoded["x_out"].F).abs().max().item())
        if max_difference != 0:
            raise RuntimeError("C2 hard encoder/decoder reconstructions differ")
        el_bits = len(encoded["strings"]) * 8
        nonzero_mask = symbols != 0
        nonzero = int(nonzero_mask.sum().item())
        active_indices = nonzero_mask.any(dim=0).nonzero(
            as_tuple=False).flatten().tolist()
        return {
            "B": B,
            "F_U": F_U,
            "D_U": D_U,
            "Full": decoded["x_out"],
            "base_bits": base_bits,
            "el_bits": el_bits,
            "full_bits": base_bits + el_bits,
            "hard_max_abs_difference": max_difference,
            "el_symbol_count": symbols.numel(),
            "el_symbol_nonzero_count": nonzero,
            "el_symbol_nonzero_fraction": nonzero / symbols.numel(),
            "el_symbol_mean_abs": float(symbols.abs().mean().item()),
            "el_symbol_min": int(symbols.min().item()),
            "el_symbol_max": int(symbols.max().item()),
            "el_active_channel_count": len(active_indices),
            "el_active_channel_indices": active_indices,
        }
