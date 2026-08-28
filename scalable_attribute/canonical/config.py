from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BaseSynthesisConfig:
    input_mode: str = "x4_f4"
    base_channels: int = 128
    block_layers: int = 2
    kernel_size: int = 3
    zero_init: bool = True

    def __post_init__(self):
        if self.input_mode not in ("x4", "x4_f4", "x4_f4_d4"):
            raise ValueError(
                "input_mode must be x4, x4_f4, or x4_f4_d4")
        for name in ("base_channels", "block_layers", "kernel_size"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls, args):
        return cls(
            input_mode=args.base_input_mode,
            base_channels=args.base_channels,
            block_layers=args.base_block_layers,
            kernel_size=args.base_kernel_size,
            zero_init=args.zero_init,
        )


def add_base_architecture_arguments(parser):
    defaults = BaseSynthesisConfig()
    parser.add_argument(
        "--base-input-mode", choices=("x4", "x4_f4", "x4_f4_d4"),
        default=defaults.input_mode)
    parser.add_argument(
        "--base-channels", type=int, default=defaults.base_channels)
    parser.add_argument(
        "--base-block-layers", type=int, default=defaults.block_layers)
    parser.add_argument(
        "--base-kernel-size", type=int, default=defaults.kernel_size)
    zero_init = parser.add_mutually_exclusive_group()
    zero_init.add_argument(
        "--zero-init", dest="zero_init", action="store_true")
    zero_init.add_argument(
        "--no-zero-init", dest="zero_init", action="store_false")
    parser.set_defaults(zero_init=defaults.zero_init)
    return parser
