# FAU HPC to N30 transfer runbook

## Default route

Large files must not be relayed through the local workstation. The tested route
is a reverse pull initiated on N30:

```text
FAU storage -> tinyx.nhr.fau.de -> csnhr.nhr.fau.de -> N30
```

N30 can reach the FAU `csnhr` SSH gateway and can then reach `tinyx` through an
SSH `ProxyCommand`. The opposite direction, FAU to the N30 public gateway,
timed out during SSH banner exchange on 2026-09-09, before authentication.
Putting the N30 key on FAU therefore does not repair that route.

## Fixed endpoints and credentials

| Role | Value |
| --- | --- |
| FAU user | `iwnt193h` |
| FAU jump host | `csnhr.nhr.fau.de:22` |
| FAU data host | `tinyx.nhr.fau.de:22` |
| Allowed FAU source root | `/home/woody/iwnt/iwnt193h/` |
| Allowed N30 destination root | `/data/run01/scz0ade/Tanzeyu/` |
| N30 FAU private key | `/data/run01/scz0ade/Tanzeyu/.ssh_transfer/id_rsa_fau_hpc` |
| Isolated host-key file | `/data/run01/scz0ade/Tanzeyu/.ssh_transfer/known_hosts` |

The credential directory must be mode 700 and the private key mode 600. Never
print or commit the private key. The user explicitly authorized installing this
FAU key on N30 on 2026-09-09.

## Routine command

Run this on N30, or invoke it through the local `paratera-n30` SSH alias:

```bash
/data/run01/scz0ade/Tanzeyu/tools/n30_pull_from_hpc.sh \
  /home/woody/iwnt/iwnt193h/scalable_attribute_thesis/datasets/example/ \
  /data/run01/scz0ade/Tanzeyu/data/example/
```

The script enforces the two allowed roots and uses `rsync --partial
--append-verify`, so an interrupted transfer can resume. A trailing slash on the
source means “copy the directory contents”; without it, rsync copies the named
file or directory itself.

For a background transfer on N30:

```bash
nohup /data/run01/scz0ade/Tanzeyu/tools/n30_pull_from_hpc.sh \
  /home/woody/iwnt/iwnt193h/scalable_attribute_thesis/datasets/example/ \
  /data/run01/scz0ade/Tanzeyu/data/example/ \
  > /data/run01/scz0ade/Tanzeyu/logs/n30_pull_example.log 2>&1 </dev/null &
```

## Verification

For material datasets, compare counts, byte sizes, and SHA-256 on both ends.
Archive downloads must also pass their native integrity test, for example:

```bash
unzip -tq dataset.zip
sha256sum dataset.zip
```

The 2026-09-09 CTC transfer moved 603,664,732 bytes at about 9.66 MB/s. All ten
ZIP SHA-256 values matched between FAU and N30; the evidence is recorded in
`docs/reproduction/UNICORN_8I_OWLII_CTC_N30_READINESS_20260909.md`.

## Failure handling

- If the transfer stops, rerun the same command; do not delete partial files.
- If `tinyx` is unreachable directly, keep the `csnhr` jump in place; direct
  access to `tinyx` from N30 is not expected.
- If authentication fails, check key permissions and expiry without displaying
  the key.
- If hashes differ, retain both copies, stop, and diagnose before extracting or
  running experiments.
