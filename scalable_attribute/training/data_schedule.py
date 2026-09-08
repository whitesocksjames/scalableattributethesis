"""Deterministic batch order for canonical staged training."""

import hashlib
import math
import os

import torch


POLICY = "global_step_epoch_randperm_v1"
FINGERPRINT_ALGORITHM = "blake2b-128-ordered-nonempty-lines-v1"


def ordered_manifest_content_fingerprint(path):
    """Fingerprint logical manifest entries in their effective order."""
    with open(path, encoding="utf-8") as handle:
        entries = [line.strip() for line in handle if line.strip()]
    digest = hashlib.blake2b(digest_size=16)
    digest.update("\n".join(entries).encode("utf-8"))
    return "{}:{}".format(FINGERPRINT_ALGORITHM, digest.hexdigest())


class ContinuationBatchSampler(torch.utils.data.Sampler):
    """Yield deterministic batches for global steps [start_step, stop_step)."""

    def __init__(self, num_samples, batch_size, start_step, stop_step, seed):
        if num_samples < 1 or batch_size < 1:
            raise ValueError("num_samples and batch_size must be positive")
        if start_step < 0 or stop_step <= start_step:
            raise ValueError("invalid continuation step range")
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.seed = int(seed)
        self.steps_per_epoch = int(math.ceil(num_samples / batch_size))

    def __len__(self):
        return self.stop_step - self.start_step

    def position(self, global_step):
        return divmod(int(global_step), self.steps_per_epoch)

    def __iter__(self):
        current_epoch = None
        permutation = None
        for global_step in range(self.start_step, self.stop_step):
            epoch, batch_in_epoch = self.position(global_step)
            if epoch != current_epoch:
                generator = torch.Generator()
                generator.manual_seed(self.seed + epoch)
                permutation = torch.randperm(
                    self.num_samples, generator=generator).tolist()
                current_epoch = epoch
            begin = batch_in_epoch * self.batch_size
            end = min(begin + self.batch_size, self.num_samples)
            yield permutation[begin:end]

    def metadata(self, manifest, data_root):
        manifest = os.path.realpath(manifest)
        return {
            "policy": POLICY,
            "seed": self.seed,
            "manifest": manifest,
            "ordered_manifest_content_fingerprint": (
                ordered_manifest_content_fingerprint(manifest)),
            "data_root_realpath": os.path.realpath(data_root),
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "drop_last": False,
            "steps_per_epoch": self.steps_per_epoch,
        }


def require_compatible_schedule(saved, current):
    """Reject a resume that cannot preserve the canonical data cursor."""
    if saved is None:
        raise ValueError(
            "Resume checkpoint predates the explicit data-order schedule; "
            "continuation order cannot be guaranteed")
    for key in (
            "policy", "seed", "manifest",
            "ordered_manifest_content_fingerprint", "data_root_realpath",
            "num_samples", "batch_size", "drop_last", "steps_per_epoch"):
        if saved.get(key) != current.get(key):
            raise ValueError(
                "Resume data-order schedule mismatch for {}: {!r} != {!r}".format(
                    key, saved.get(key), current.get(key)))
