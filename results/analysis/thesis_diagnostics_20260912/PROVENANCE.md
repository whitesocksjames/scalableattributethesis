# Provenance

This branch was created from `main` commit
`597059b1c1a65cde5b7c32faddeffc25a85f6b79`. The formal evidence source is
`results/comparisons/formal_static_rgb_final_20260911`; it remains unchanged.

The 256 analysis uses the historically established sequential diagnostic
lineage (Base `2k128 / lambda256 / step3525`, Enhancement Stage-1 `step1763`).
The 8iVFB/Owlii measurements were retained historical physical results and the
12 missing CTC measurements used the frozen formal input, chunk, metric, and
physical-bit semantics. The 256 point remains outside the formal curve.

Q3 selected 13 frozen negative YUV611 cases and 16 deterministic matched
controls. Diagnostics were computed from coordinate-key-aligned,
pre-quantization Base/Full tensors using each checkpoint's recorded D111
objective. Twenty-eight cases passed exact frozen bits, reconstruction SHA, and
metric gates. Redandblack 8K differs by -16 Base and Full bits, retains the exact
Enhancement payload, and differs by at most 0.0003 dB; it is never labeled exact.

No checkpoint, reconstruction PLY, bitstream, raw cluster output, scheduler
log, credential, account identifier, or private infrastructure path is included
in this package.
