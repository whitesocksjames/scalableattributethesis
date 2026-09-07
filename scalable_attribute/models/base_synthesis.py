import MinkowskiEngine as ME
import torch

from basic_models.backbone import Backbone


class BaseSynthesis(torch.nn.Module):
    """One Unicorn-style stride-2 to stride-1 learned compensation path."""

    _INPUT_CHANNELS = {
        "x4": 3,
        "x4_f4": 3 + 128,
        "x4_f4_d4": 3 + 128 + 128,
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = Backbone(
            scale=-1,
            in_channels=self._INPUT_CHANNELS[config.input_mode],
            channels=config.base_channels,
            out_channels=128,
            block_type="resnet",
            block_layers=config.block_layers,
            kernel_size=config.kernel_size,
            stride=[2, 2, 2],
        )
        if config.zero_init:
            for parameter in self.backbone.linear_out.parameters():
                torch.nn.init.zeros_(parameter)

    def forward(self, state):
        values = {
            "x4": [state.x4],
            "x4_f4": [state.x4, state.f4],
            "x4_f4_d4": [state.x4, state.f4, state.d4],
        }[self.config.input_mode]
        return self.backbone(ME.cat(values))
