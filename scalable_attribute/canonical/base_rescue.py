"""Small deterministic primitives for Base rescue screening."""

import csv
import hashlib
import math
from pathlib import Path

import torch

from basic_models.loss import get_bits


SAMPLER_POLICY = "global_draw_weighted_replacement_v1"


def sample_key(path):
    parts = Path(path).parts
    if len(parts) < 2:
        raise ValueError("Cannot form stable sample key: " + str(path))
    return "/".join(parts[-2:])


def load_difficulty_scores(path, score_column="r2_E_D111"):
    scores = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = sample_key(row["source"])
            if key in scores:
                raise ValueError("Duplicate difficulty key: " + key)
            scores[key] = float(row[score_column])
    if not scores:
        raise ValueError("Difficulty table is empty")
    return scores


def classify_difficulty(files, scores):
    """Return exact bottom/middle/top quartile labels with stable tie breaks."""
    ranked = []
    for index, path in enumerate(files):
        key = sample_key(path)
        if key not in scores:
            raise KeyError("No difficulty score for " + key)
        ranked.append((scores[key], key, index))
    ranked.sort()
    quartile = int(math.ceil(len(ranked) * 0.25))
    labels = ["normal"] * len(files)
    for _, _, index in ranked[:quartile]:
        labels[index] = "low"
    for _, _, index in ranked[-quartile:]:
        labels[index] = "high"
    return labels, {
        "score": "GT r2 residual D111 energy",
        "low_count": quartile,
        "normal_count": len(files) - 2 * quartile,
        "high_count": quartile,
        "low_max": ranked[quartile - 1][0],
        "high_min": ranked[-quartile][0],
        "tie_break": "score_then_last_two_path_components",
    }


class DeterministicWeightedBatchSampler(torch.utils.data.Sampler):
    """Deterministic with-replacement draws for uniform/3x-high policies."""

    def __init__(self, labels, batch_size, steps, seed, high_weight):
        if batch_size < 1 or steps < 1 or high_weight < 1:
            raise ValueError("Invalid weighted sampler configuration")
        self.labels = tuple(labels)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.seed = int(seed)
        self.high_weight = float(high_weight)
        weights = torch.ones(len(labels), dtype=torch.double)
        weights[torch.tensor([x == "high" for x in labels])] = high_weight
        generator = torch.Generator().manual_seed(seed)
        self.draws = torch.multinomial(
            weights, batch_size * steps, replacement=True,
            generator=generator).tolist()

    def __len__(self):
        return self.steps

    def __iter__(self):
        for begin in range(0, len(self.draws), self.batch_size):
            yield self.draws[begin:begin + self.batch_size]

    def metadata(self, files):
        digest = hashlib.blake2b(digest_size=16)
        digest.update("\n".join(files[i] for i in self.draws).encode("utf-8"))
        counts = {name: 0 for name in ("low", "normal", "high")}
        for index in self.draws:
            counts[self.labels[index]] += 1
        return {
            "policy": SAMPLER_POLICY,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "replacement": True,
            "high_weight": self.high_weight,
            "draw_count": len(self.draws),
            "actual_draw_counts": counts,
            "sampled_order_blake2b128": digest.hexdigest(),
        }


def base_rescue_objective(output, attribute, lmb):
    likelihoods = output.get("prefix_likelihoods")
    if len(likelihoods or []) != 4:
        raise RuntimeError("Base rescue rate must contain exactly r1-r4")
    rate = sum(get_bits(value) for value in likelihoods) / len(attribute)
    channel_mse = torch.mean((attribute.F - output["Base"].F) ** 2, dim=0)
    distortion = channel_mse.mean()
    return rate + lmb * distortion, rate, distortion, channel_mse
