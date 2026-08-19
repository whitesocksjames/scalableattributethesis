import torch


class ScalableAttributeCoder:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def test(self, A, base_lambda):
        self.model.eval()
        B, F_U, base_bits = self.model.base_adapter.hard_reconstruct(A, base_lambda)
        encoded, Full_encoder = self.model.enhancement.hard_encode(A, B, F_U)
        Full = self.model.enhancement.hard_decode(encoded, B, F_U)
        if not torch.equal(Full_encoder.C, Full.C):
            raise RuntimeError("EL hard encoder/decoder coordinates differ")
        max_difference = (Full_encoder.F - Full.F).abs().max().item()
        if max_difference != 0:
            raise RuntimeError("EL hard encoder/decoder reconstructions differ")

        el_bits = len(encoded.strings) * 8
        num_points = len(A)
        return {
            "B": B,
            "F_U": F_U,
            "Full": Full,
            "base_bits": base_bits,
            "el_bits": el_bits,
            "full_bits": base_bits + el_bits,
            "R_base": base_bits / num_points,
            "R_E": el_bits / num_points,
            "R_full": (base_bits + el_bits) / num_points,
            "hard_max_abs_difference": max_difference,
        }
