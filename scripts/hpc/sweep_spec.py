"""Single editable entry point for thesis sweeps.

Keep this file empty until a real experiment is agreed. Values shown in comments
are field names, not formal thesis settings.
"""

from sweep_utils import grid


# Official V1's object test pairs 32k8k with 32768/16384/8192, 8k256 with
# 8192/4096/2048, and 2k128 with 1024/512/256/128. These are overlapping
# variable-rate profiles, not a unique lambda-to-file rule. The released package
# currently available in the thesis workspace contains only 2k128, and the real
# RWTT interface smoke also validated it at lambda=2048. Therefore the thesis
# default names that verified combination without restricting explicit overrides.
BASE_CHECKPOINT_PROFILES = {
    "thesis_rwtt_2k128": {
        "checkpoint": (
            "unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/"
            "lossy_attribute/rwtt/2k128/epoch_last.pth"
        ),
        "validated_lambda": 2048,
    },
}
DEFAULT_BASE_PROFILE = "thesis_rwtt_2k128"

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
        "model_95_5_seed0/val_h5.txt"
    ),
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
EVAL_EXPERIMENTS = []
