# Diagnostic 256 analysis — 2026-09-12

This is private, post-freeze diagnostic evidence. It does not alter the frozen
seven-point curve, formal BD-BR, checkpoint selection, or public result package.

## Verified facts

- The audited sequential lineage is complete for 20/20 samples: 4 8iVFB,
  4 Owlii, and 12 CTC.
- Every Enhancement payload is exactly one byte per independently coded input:
  8 bits for each unchunked 8iVFB/Owlii sample and 8 bits per CTC chunk. CTC
  sample totals therefore range from 8 to 64 bits solely with chunk count.
- Across 20 samples, raw Full-minus-Base YUV611 signs are 9 positive and
  11 negative. With the predeclared 0.01 dB near-zero band, 8 are positive,
  6 near-zero, and 6 negative.
- The predeclared universal-collapse decision is `NO claim`: only 6/20 samples
  satisfy `|Delta YUV611| <= 0.01 dB`.

| Dataset | Samples | Mean Enhancement bits | Mean Delta bpp | Mean Delta YUV611 | Positive >0.01 | Near-zero | Negative <-0.01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8iVFB | 4 | 8 | 0.00000929 | -0.00675 dB | 0 | 3 | 1 |
| Owlii | 4 | 8 | 0.00001150 | -0.01112 dB | 0 | 2 | 2 |
| CTC | 12 | 35.33 | 0.00001295 | +0.04354 dB | 8 | 1 | 3 |

## Matched 512 control

| Dataset | 256 mean Delta YUV611 / Delta bpp | 512 mean Delta YUV611 / Delta bpp |
| --- | --- | --- |
| 8iVFB | -0.00675 dB / 0.00000929 | +0.37678 dB / 0.01284871 |
| Owlii | -0.01112 dB / 0.00001150 | +0.23852 dB / 0.00370762 |
| CTC | +0.04354 dB / 0.00001295 | +0.08266 dB / 0.00108833 |

The existing RWTT-28Lite validation trajectory is consistent with this
boundary: at step 1763, 256 has Enhancement rate 0.0001093 bpp and changes
YUV611 by about -0.0096 dB, whereas matched 512 has 0.0335965 bpp and improves
YUV611 by about +0.4725 dB.

## Supported interpretation

The 256 entropy-coded Enhancement payload reaches the observed one-byte coder
floor for every independently coded input. This is strong evidence of a
near-zero-rate Enhancement. It is not evidence that Full always equals Base:
with effectively no content-adaptive payload, the conditional decoder can still
produce a deterministic Base-dependent correction, and CTC shows several
measurable gains as well as losses.

The defensible boundary claim is therefore dataset-sensitive: 256 provides no
reliable scalable separation on 8iVFB/Owlii and is unstable across CTC, while
512 consistently allocates nontrivial Enhancement rate and is positive on
19/20 samples under the 0.01 dB classification.

## Remaining uncertainty

- Quantized latent non-zero counts and symbol diversity were not persisted.
  One-byte-per-input physical streams support a coder-floor statement, but do
  not alone prove the exact latent symbol distribution.
- Training logs expose noisy estimated rate and Full distortion but not a
  matched per-batch Base distortion, so they cannot prove that RD-objective
  balance caused the observed outcome.

## Thesis-safe wording

> The post-freeze 256 diagnostic used the historically established sequential
> lineage. Its Enhancement stream reached the observed one-byte-per-input coder
> floor. Base-to-Full quality changes were negligible or negative on 8iVFB and
> Owlii but content-dependent on CTC, whereas the matched 512 point allocated
> nontrivial Enhancement rate and produced substantially more consistent gains.
> This supports 512 as the lower boundary of the frozen formal scalable range.

Do not claim universal latent collapse, universal Full/Base equality, or that
256 was excluded after inspecting CTC.

## Artifacts

- `tables/diagnostic256_vs_512.csv`
- `tables/diagnostic256_dataset_summary.csv`
- `figures/diagnostic256_full_minus_base.svg`
