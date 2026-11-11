# Jianqiang Wang (wangjq@smail.nju.edu.cn)
# Last update: 2024-01-10

"""Fixed configuration for the released attribute inference model.

The checkpoint architecture is part of the evaluation contract.  Keeping these
values here avoids importing training and dataset configuration or parsing the
host process's command line when the model is imported.
"""

from types import SimpleNamespace


def get_args(component="attribute"):
    """Return the immutable-by-convention inference settings for attributes."""
    if component != "attribute":
        raise ValueError("Only the released attribute model is supported")
    return SimpleNamespace(
        scale=5,
        stage=1,
        Vmode=1,
        channels=128,
        latent_channels=32,
        in_channels=3,
        block_type="resnet",
        block_layers=2,
        kernel_size=3,
        skip_mode=0,
        color_format="yuv",
        DBG=0,
        normalize=1,
    )


if __name__ == "__main__":
    print(get_args())
