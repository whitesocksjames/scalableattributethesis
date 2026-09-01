# Canonical Repository and Artifact Cleanup Plan

Status: non-destructive planning only. No item in this document authorizes a
delete, move, rename, overwrite or recursive cleanup.

## Recovery point

- Git commit: `150ef658e429c215b549be74e2d73e2a631c9527`
- Tag: `canonical-evidence-snapshot-20260901`
- Remote: `https://github.com/whitesocksjames/scalableattributethesis.git`
- Remote branch and tag were verified before this plan was created.

## Goal

Keep the active repository easy to navigate: clean official Unicorn baseline,
canonical thesis runtime, dataset/reference infrastructure, cluster guides and
a small number of evidence documents. Historical implementation remains
recoverable through Git and frozen experiment artifacts.

## Classification

### KEEP ACTIVE

- Official Unicorn source and the minimal compatibility patches still needed
  by N30/HPC.
- `scalable_attribute/canonical/` and canonical training/evaluation entrypoints.
- RWTT/MVUB preprocessing adapters and manifests.
- Official checkpoint mapping and reference evaluation/aggregation code.
- `HPC.md`, `HPC_GUIDE.md`, `N30_GUIDE.md`, `DEVELOPMENT_RULES.md`.
- Compact final figures, their plotting scripts and machine-readable source
  tables.

### KEEP AS EVIDENCE, NOT ACTIVE RUNTIME

- Architecture survey/screening reports.
- External-EL V1, C2, B1 and other rejected/frozen research evidence.
- Direct-D611, D411/D611, MVUB and Base-D611 ablation summaries needed to
  justify decisions.
- Raw CSV/JSON necessary to reproduce reported tables and figures.

### CLEANUP CANDIDATE AFTER EXACT REVIEW

- Duplicate copied logs and repeated rendered figures.
- Superseded checkpoints that are neither selected nor required to resume a
  documented experiment.
- Experiment-local GPCC/torchac temporary files and disposable `/dev/shm`
  staging directories belonging to a finished job.
- Obsolete active runtime already recoverable from the evidence tag, but only
  after verifying that no current evaluator imports it.

### NEVER CLEAN AS PART OF THIS PHASE

- Datasets, released checkpoints, selected canonical checkpoints.
- Formal `per_h5.csv`, `per_model.csv`, curve/endpoint CSV and provenance JSON.
- Anything outside `/data/run01/scz0ade/Tanzeyu/` on N30.
- Any directory whose ownership, dependency or recovery status is uncertain.

## Safe sequence

1. Freeze source/evidence in Git (complete).
2. Generate protocol, plan and candidate inventory (this phase).
3. Resolve every candidate to an exact path and measure its size without broad
   traversal or deletion.
4. For each candidate, verify current import references and identify its
   replacement/recovery source.
5. Present a deletion/move list for explicit manager approval.
6. Perform only exact approved actions; no wildcard deletion, `find -delete`,
   `rsync --delete`, automatic cache cleaner or broad recursive command.
7. Run baseline import, checkpoint-load, H5-read and evaluator regression gates.
8. Commit the cleanup separately from canonical model changes.

## N30 rules

All persistent reads/writes and any future cleanup are restricted to:

```text
/data/run01/scz0ade/Tanzeyu/
```

The only external namespace permitted by project policy is a job-local
temporary directory created by that job:

```text
/dev/shm/Tanzeyu_${SLURM_JOB_ID}/
```

Only that exact job-local directory may be removed by the job. Do not inspect
or manipulate other shared-account users' directories. Existing N30 experiment
directories should not be rearranged merely for aesthetics because stored
commands and checkpoint provenance may contain their absolute paths.

## Output organization after cleanup

Do not create a database or duplicate experiment tree. Add a compact index that
points to existing immutable artifacts:

```text
canonical final candidates
canonical ablations/evidence
formal reference results
figures + source CSV
```

Each selected checkpoint entry should record cluster, exact path, operating
point, initialization lineage, training protocol, and matching evaluation
artifact. This is a pointer/index, not a copy.

## Regression gates after a future physical cleanup

- released Unicorn checkpoint load;
- official reference forward/hard decode;
- RWTT H5 read and DataLoader;
- canonical Base/Full checkpoint load;
- canonical hard bit identities;
- formal aggregation golden-input regression;
- N30 training entrypoint import;
- `git diff --check`.

## Deferred decisions

- No physical N30 artifact deletion is approved.
- No source file is approved for deletion by this plan alone.
- Consolidating evaluation scripts is a separate refactor and must preserve
  golden CSV output before old entrypoints are retired.
- Final multi-rate composite checkpoint packaging waits for lower-rate
  Enhancement stage-gate results.

