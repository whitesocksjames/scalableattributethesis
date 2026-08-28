from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnhancementConfig:
    attribute_channels: int = 3
    base_feature_channels: int = 128
    hidden_channels: int = 128
    latent_channels: int = 64
    analysis_scale: int = 2
    kernel_size: int = 3
    block_layers: int = 2
    block_type: str = "resnet"
    synthesis_condition: str = "b_fu"
    zero_centered_synthesis: bool = False

    def __post_init__(self):
        if self.synthesis_condition not in ("b_fu", "b", "none"):
            raise ValueError(
                "synthesis_condition must be b_fu, b, or none")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls, args):
        return cls(**{name: getattr(args, name) for name in cls.__dataclass_fields__})


def add_architecture_arguments(parser):
    defaults = EnhancementConfig()
    parser.add_argument("--attribute-channels", type=int, default=defaults.attribute_channels)
    parser.add_argument("--base-feature-channels", type=int, default=defaults.base_feature_channels)
    parser.add_argument("--hidden-channels", type=int, default=defaults.hidden_channels)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--analysis-scale", type=int, default=defaults.analysis_scale)
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument("--block-layers", type=int, default=defaults.block_layers)
    parser.add_argument("--block-type", default=defaults.block_type)
    parser.add_argument(
        "--synthesis-condition", choices=("b_fu", "b", "none"),
        default=defaults.synthesis_condition)
    parser.add_argument(
        "--zero-centered-synthesis", action="store_true",
        default=defaults.zero_centered_synthesis)
    return parser
