"""Official Unicorn-v1 RWTT Attribute nine-point reference curve."""

from scalable_attribute.reference_points import (
    OFFICIAL_RWTT_REFERENCE_POINTS, reference_checkpoint)


REFERENCE_COMMON = {
    "file_list": (
        "$WORK/scalable_attribute_thesis/datasets/RWTT/splits/"
        "model_95_5_seed0/val_h5.txt"
    ),
}

REFERENCE_EXPERIMENTS = [
    {
        "name": rate_id,
        "rate_id": rate_id,
        "checkpoint_profile": checkpoint_profile,
        "base_lambda": base_lambda,
        "base_checkpoint": reference_checkpoint(checkpoint_profile),
    }
    for rate_id, checkpoint_profile, base_lambda
    in OFFICIAL_RWTT_REFERENCE_POINTS
]
