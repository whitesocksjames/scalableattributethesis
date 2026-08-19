import torch

from scalable_attribute.base_adapter import BaseAdapter
from scalable_attribute.enhancement_layer import ConditionalEnhancementLayer


class ScalableAttributeModel(torch.nn.Module):
    def __init__(self, base_checkpoint, enhancement_config, base_scale=5, base_stage=1, base_vmode=1):
        super().__init__()
        self.base_adapter = BaseAdapter(
            base_checkpoint,
            scale=base_scale,
            stage=base_stage,
            vmode=base_vmode,
            attribute_channels=enhancement_config.attribute_channels,
            feature_channels=enhancement_config.base_feature_channels,
        )
        self.enhancement = ConditionalEnhancementLayer(enhancement_config)

    def forward(self, A, base_lambda):
        B, F_U = self.base_adapter(A, base_lambda)
        output = self.enhancement(A, B, F_U)
        output["B"] = B
        output["F_U"] = F_U
        return output
