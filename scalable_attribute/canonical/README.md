# Compatibility package

`scalable_attribute.canonical` preserves historical imports and recorded
experiment commands. It is not a second implementation of the model.

The preferred reading order for current code is:

1. `scalable_attribute/models/` — Prefix, BaseSynthesis, Base reconstruction,
   and independent Enhancement architecture;
2. `scalable_attribute/training/` — optimization and trainable-scope policy
   (introduced in later reviewed batches);
3. `scalable_attribute/evaluation/` — metrics and evaluation logic;
4. `scalable_attribute/runtime/` — strict checkpoint loading and runtime
   support.

Files in this directory are compatibility re-exports until their corresponding
implementation has moved. New architecture code should not be added here.
