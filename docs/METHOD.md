# Method

The method adds two full-resolution decoding endpoints to the Unicorn Part II attribute codec:

```text
truncated Unicorn prefix -> BaseSynthesis -> full-resolution Base
full-resolution Base + conditional Enhancement -> Full
```

## Base endpoint

The Base bitstream contains the native `x_low` attribute payload and residual payloads `r1-r4`; it omits `r5`. Decoding the prefix exposes low-resolution attributes and intermediate features. BaseSynthesis maps those decoded features into compensation for the full-resolution reconstruction path. Native fusion and output networks combine this compensation with the full-resolution feature path to produce Base. BaseSynthesis adds no payload bits.

## Full endpoint

The Full bitstream contains the Base payload plus a separately entropy-coded Enhancement payload. Enhancement decoding is conditioned on decoded Base state. Full decoding reuses the Base reconstruction and applies the decoded enhancement residual.

## Training objective

The Base objective combines estimated `r1-r4` prefix rate with Base distortion. The Full objective adds Enhancement rate and Full distortion:

```text
L_Base = R_Base + lambda_Base * D_Base
L_Full = R_Base + R_Enhancement + lambda_Full * D_Full
```

The fixed `x_low` rate is independent of learned parameters and is omitted from the differentiable training-rate term. Formal physical-rate measurements include it. The method builds on the Unicorn Part II implementation and adds BaseSynthesis and the conditional Enhancement endpoint.

The seven final operating points are `512`, `1k`, `2k`, `4k`, `8k`, `16k`, and `32k`. Channel-average distortion uses equal Y/U/V weights for the main training objective; the selected 32k Enhancement model uses Y/U/V weights 6/1/1. The handover supplies frozen pretrained inference states. Training runs and checkpoint selection history remain in the research archive.

The exported states preserve the selected final optimization scope. At 2k, the Base file restores both the tuned Prefix and BaseSynthesis before the selected Enhancement state is added. At 4k, a complete jointly fine-tuned Base/Enhancement state replaces its 8k initialization. The other points combine the released Prefix with the selected learned BaseSynthesis and Enhancement states. These restoration differences retain the same Base/Full architecture and payload definitions; all seven published files contain the complete inference graph.
