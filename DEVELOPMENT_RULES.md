# Development Rules

- This repository is a Master Thesis research prototype, not a production system.
- Avoid overengineering.
- Do not add SHA256 verification, file locking, transactions, complex caches, complex recovery logic, repeated audits, or excessive defensive checks.
- Keep validation limited to what is necessary for experimental correctness:
  - tensor shapes and coordinate alignment;
  - finite loss and rate values;
  - basic forward-backward execution;
  - hard encode/decode execution;
  - final bitrate and reconstruction correctness.
- Do not add large wrapper layers or test frameworks in the name of safety.
- Do not modify geometry.
- Do not modify the Unicorn Base architecture or entropy model unless explicitly requested.
- Scalable attribute coding is the only official research direction in this repository.
