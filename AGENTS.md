# Project operating rules

These repository-specific rules override broader Codex orchestration defaults
for work inside this repository.

## HPC and Slurm turn boundaries

- Deterministic submission-time or infrastructure failures, including job-count
  limits, invalid resource requests, launch or environment errors, missing
  runtime paths, and scheduler rejection, may be diagnosed, corrected, and
  resubmitted in the same turn when the scientific contract is unchanged.
- Keep corrective submission retries bounded, normally to at most two or three
  attempts. Never enter an indefinite retry or polling loop.
- Scientific correctness failures, dataset or checkpoint identity failures,
  metric or rate inconsistencies, and any correction that would change the
  scientific contract must stop for user review.
- After a job is accepted, perform one immediate status check and record its job
  ID. Do not continuously poll a `PENDING` or `RUNNING` job.
- For `PENDING`, report the state and end the current turn.
- For `RUNNING`, an ETA may be estimated from existing runtime evidence without
  more scheduler queries. Only when the estimated remaining runtime is roughly
  two minutes or less may one additional bounded status check be made; otherwise
  report and end the turn.
- Never use `sleep` plus a polling loop, and never repeatedly call `squeue`,
  `sacct`, or SSH merely to wait for a job to finish.
- Never spawn a subagent to monitor a scheduler.
- Query a previously submitted job again only when the user explicitly asks for
  a status check, and then query it once.
- Never keep a Codex turn active for tens of minutes waiting for an HPC job.
- Token and usage cost cannot be predicted precisely. Use only a coarse
  low/medium/high estimate when deciding whether an immediate bounded follow-up
  is worthwhile.

## Delegation limits

- Spawn a subagent only for work that is genuinely independent, clearly
  parallelizable, and requires a substantial repository scan or similarly
  bounded body of work.
- Do not spawn for ordinary SSH queries, Slurm status checks, one log or file
  read, simple validation, or a short edit.
- Prefer one agent for small tasks. Do not delegate merely to create parallel
  activity.

## Formal experiment release safety

- Without explicit user authorization, do not automatically progress from a
  preflight or gate to releasing the next formal job batch.
- Authorization to finish a finite formal matrix also authorizes mechanical
  continuation of later batches split only by scheduler or job-count limits.
- When a later explicit status check shows capacity is available, submit the
  next already-authorized mechanical batch directly; do not require new approval
  for every such batch.
- After each accepted batch submission, report the job IDs, purpose, and output
  root, then follow the HPC turn-boundary rules above.

## Formal CTC packing

- Prefer sample-level Slurm packing for formal CTC evaluation.
- One sample job may serially execute its remaining Original Unicorn and Ours
  operating points.
- Every operating point must retain an independent subprocess, status, output,
  and provenance record. One point failure must not prevent subsequent valid
  points from running.
- Avoid method-wide jobs spanning many samples unless a concrete scheduler
  constraint requires them.

## HPC data transfers

- Do not relay large payloads through the local workstation.
- Use the project transfer runbook and its documented reverse-pull route.
- Use `scripts/transfer/n30_pull_from_hpc.sh` for routine transfers.
- Never print, commit, or copy credentials into logs.
- Use resumable transfer and verify material datasets with size plus SHA-256 at
  both endpoints.
