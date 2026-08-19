import torch


def rate_distortion_loss(A, Full, likelihood, rd_lambda):
    num_points = len(A)
    rate = -torch.log2(likelihood).sum() / num_points
    distortion = torch.mean((A.F - Full.F) ** 2)
    loss = rate + rd_lambda * distortion
    if not torch.isfinite(torch.stack([loss, rate, distortion])).all():
        raise RuntimeError("Non-finite EL loss, rate or distortion")
    return loss, rate, distortion
