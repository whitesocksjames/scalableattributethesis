# Project operating rules

These repository-specific rules override broader Codex orchestration defaults
for work inside this repository.

## HPC and Slurm turn boundaries

- After `sbatch`, perform at most one immediate status check and record the job
  ID.
- If the submitted job is `PENDING` or `RUNNING`, report its state and end the
  current turn immediately.
- Never use `sleep` plus a polling loop, and never repeatedly call `squeue`,
  `sacct`, or SSH merely to wait for a job to finish.
- Never spawn a subagent to monitor a scheduler.
- Query a previously submitted job again only when the user explicitly asks for
  a status check, and then query it once.
- Never keep a Codex turn active for tens of minutes waiting for an HPC job.

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
- After each batch submission, report the job IDs, purpose, and output root,
  then end the current turn. A later batch requires a subsequent explicit user
  instruction.

## HPC data transfers

- Do not relay large payloads through the local workstation.
- Use the project transfer runbook and its documented reverse-pull route.
- Use `scripts/transfer/n30_pull_from_hpc.sh` for routine transfers.
- Never print, commit, or copy credentials into logs.
- Use resumable transfer and verify material datasets with size plus SHA-256 at
  both endpoints.
