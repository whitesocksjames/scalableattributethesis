# Joint 4K All-Checkpoint RD Review

All values use existing physical hard evaluations. Official deltas use adjacent-point interpolation only; no extrapolation.

## Seven-checkpoint summary

| Step | Base in range | Full in range | Base Δofficial 8i / Owlii | Full Δofficial 8i / Owlii | Mean / max Base vs 8K rate | Pareto violations B/F | Enhancement RD+ | Negative Full gain |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0/8 | 8/8 | -0.161 / -0.662 | -0.218 / -0.459 | +21.2% / +25.3% | 8/8 | 5/8 | 0/8 |
| 1000 | 0/8 | 8/8 | -0.008 / -0.490 | +0.051 / -0.324 | +20.9% / +27.3% | 7/6 | 6/8 | 0/8 |
| 1500 | 0/8 | 8/8 | -0.076 / -0.474 | +0.014 / -0.245 | +20.1% / +27.4% | 4/1 | 7/8 | 0/8 |
| 2000 | 0/8 | 8/8 | -0.158 / -0.415 | -0.128 / -0.272 | +18.5% / +24.3% | 4/4 | 7/8 | 0/8 |
| 2500 | 0/8 | 8/8 | -0.247 / -0.595 | -0.290 / -0.547 | +20.4% / +24.6% | 8/8 | 2/8 | 1/8 |
| 3000 | 0/8 | 8/8 | +0.059 / -0.365 | +0.066 / -0.238 | +17.8% / +22.1% | 0/1 | 5/8 | 1/8 |
| 3525 | 0/8 | 8/8 | -0.059 / -0.449 | +0.031 / -0.272 | +22.6% / +29.5% | 3/2 | 6/8 | 0/8 |

## Screening interpretation

- `STRICT PASS` requires Base and Full rates inside the frozen 2K–8K brackets on all sequences, no Pareto violation, positive Enhancement RD gain on all sequences, and all correctness gates.
- `USABLE CANDIDATE` may retain a Base placement warning when its RD remains competitive and it is not systematically dominated.
- `REJECT` denotes clear multi-sequence Pareto domination, ineffective Full refinement, or correctness failure.

The numerical shortlist and trade-offs are reported after inspecting this table; no checkpoint is automatically made canonical.

## Findings

- Base-rate overshoot is present from `step500`; it is not a late-training drift. Mean overshoot versus the selected 8K Base stays between `+17.8%` and `+22.6%`, with no monotonic rise from step500 to final.
- Every Full checkpoint remains rate-bracketed between the frozen 2K and 8K Full endpoints on all eight sequences.
- `step2500` is rejected: it has eight Base and eight Full Pareto violations, only 2/8 positive Enhancement RD gains, and one negative Full quality increment.
- `step500`, `step1000`, and `step2000` are not competitive with the later frontier. `step3525` increases Base overshoot and is not better than the strongest earlier checkpoints.
- No checkpoint is a `STRICT PASS`, because Base rate is above the selected 8K Base on all eight sequences.

## Shortlist for manager review

### step3000 — USABLE CANDIDATE

- Best Base behavior: zero Base Pareto violations and the smallest mean/max Base-rate overshoot (`+17.8% / +22.1%`).
- Best mean Full matched-rate result across the combined evidence: `+0.066 dB` on 8i and `-0.238 dB` on Owlii.
- One Full Pareto violation and one negative Full increment occur on Loot (`-0.0147 dB`), so it is not a clean strict pass.

### step1500 — USABLE CANDIDATE

- Full endpoint is highly competitive: only one Full Pareto violation, `+0.014 dB` on 8i and `-0.245 dB` on Owlii.
- Enhancement RD gain is positive on 7/8 sequences, the best count in the trajectory.
- Base is weaker than step3000: four Base Pareto violations and slightly larger mean rate overshoot (`+20.1%`).

## Overall classification

The joint transfer produces a healthy **Full 4K operating region**, but not a clean Base-rate ladder. The Base overshoot is structural to this joint trajectory from its earliest saved checkpoint rather than evidence that step3000 alone trained too long. The two evidence-supported choices are therefore step3000 for the strongest joint Base/Full compromise and step1500 for the most consistently useful Enhancement refinement. Final selection remains a manager decision.
