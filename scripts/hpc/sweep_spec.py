"""Single editable entry point for thesis sweeps.

Keep this file empty until a real experiment is agreed. Values shown in comments
are field names, not formal thesis settings.
"""

from sweep_utils import grid
from scalable_attribute.reference_points import (
    OFFICIAL_RWTT_REFERENCE_POINTS, reference_checkpoint)


# Official V1's object test pairs 32k8k with 32768/16384/8192, 8k256 with
# 8192/4096/2048, and 2k128 with 1024/512/256/128. These are overlapping
# variable-rate profiles, not a unique lambda-to-file rule. The released package
# currently available in the thesis workspace contains only 2k128, and the real
# RWTT interface smoke also validated it at lambda=2048. Therefore the thesis
# default names that verified combination without restricting explicit overrides.
BASE_CHECKPOINT_PROFILES = {
    rate_id: {
        "checkpoint": reference_checkpoint(checkpoint_profile),
        "validated_lambda": base_lambda,
    }
    for rate_id, checkpoint_profile, base_lambda
    in OFFICIAL_RWTT_REFERENCE_POINTS
}
# Preserve already-running checkpoints exactly, but never select this profile
# implicitly for a future experiment.
BASE_CHECKPOINT_PROFILES["legacy_running_2k128_l2048"] = {
    "checkpoint": reference_checkpoint("2k128"),
    "validated_lambda": 2048,
    "legacy": True,
}
DEFAULT_BASE_PROFILE = None

# Naming is an experiment concern, not scheduler logic. Keep only fields that
# distinguish runs at a glance; resolved_args.json retains every CLI value.
EXPERIMENT_NAME_FIELDS = (
    ("base_lambda", "b"),
    ("rd_lambda", "rd"),
    ("lr", "lr"),
    ("latent_channels", "cz"),
)

# Evaluation can inherit selected values from train/resolved_args.json. Change
# this list in the spec when the model CLI evolves; HPC code forwards it blindly.
EVAL_INHERIT_TRAIN_ARGS = (
    "base_checkpoint", "base_lambda", "base_scale", "base_stage", "base_vmode",
    "rd_lambda", "lr", "seed",
)


# Architecture defaults live only in scalable_attribute/config.py. Put an
# architecture override here only when the whole study intentionally changes it.
ARCHITECTURE = {
    "latent_channels": 64,
}

# Optional shared CLI overrides. Add values only after choosing an experiment.
# Merge ARCHITECTURE here when a study needs shared architecture overrides:
# TRAIN_COMMON = {**ARCHITECTURE, ...}
TRAIN_COMMON = {
    **ARCHITECTURE,
    # Provenance for the already-running 2026-08-19 coarse sweep. New studies
    # must replace this with one of the official R01--R09 profiles and use its
    # corresponding base_lambda.
    "base_profile": "legacy_running_2k128_l2048",
    "base_lambda": 2048,
    "batch_size": 4,
    "num_workers": 0,
    "seed": 0,
    "epochs": 1,
    "train_file_list": (
        "$WORK/scalable_attribute_thesis/datasets/RWTT/splits/"
        "model_95_5_seed0/train_h5.txt"
    ),
    "val_file_list": (
        "$WORK/scalable_attribute_thesis/datasets/RWTT/splits/"
        "model_95_5_seed0/val_h5.txt"
    ),
}
EVAL_COMMON = {
    "file_list": (
        "$WORK/scalable_attribute_thesis/datasets/RWTT/splits/"
        "model_95_5_seed0/dev_val_models14_uniform_v1_h5.txt"
    ),
    "base_profile_label": "legacy_running_2k128_l2048",
}

# Original one-epoch coarse screening, now with the fixed BS4 that passed the
# largest-four-block V100 memory gate.
TRAIN_COMMON.update({
    "max_steps": 0,
    "save_every": 705,
    "val_every": 705,
})
TRAIN_EXPERIMENTS = grid(
    common=TRAIN_COMMON,
    rd_lambda=[350, 1000, 2500],
    lr=[1e-4, 5e-5],
)
EVAL_EXPERIMENTS = [
    {
        "train_experiment": "b2048_rd350_lr0p0001_cz64",
        "checkpoint_tag": "step3525",
        "base_profile": "legacy_running_2k128_l2048",
    },
    {
        "train_experiment": "b2048_rd350_lr5e-05_cz64",
        "checkpoint_tag": "step3525",
        "base_profile": "legacy_running_2k128_l2048",
    },
    {
        "train_experiment": "b2048_rd1000_lr0p0001_cz64",
        "checkpoint_tag": "step3525",
        "base_profile": "legacy_running_2k128_l2048",
    },
    {
        "train_experiment": "b2048_rd1000_lr5e-05_cz64",
        "checkpoint_tag": "step3525",
        "base_profile": "legacy_running_2k128_l2048",
    },
]
