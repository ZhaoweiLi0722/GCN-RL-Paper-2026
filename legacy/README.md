# Legacy Research Code

This folder contains source code imported from the previous capacity-planning paper archive:

`DRL for capacity planning considering patient information (1).zip`

The code is preserved for reference and migration. It is not treated as the canonical implementation for the current GCN-DDPG project.

Imported items:

- `cp_decentralized/`: previous decentralized capacity-planning source code.
- `../docs/previous_work/DynRes.pdf`: reference paper/document.
- `../docs/previous_work/GNN-RL-Knowledge-based.docx`: reference notes/document.

Excluded items:

- macOS metadata such as `.DS_Store` and `__MACOSX/`.
- model checkpoints such as `.pth` files.

Migration notes:

- The legacy scripts use older local imports and external packages such as `gym`, `torch`, `scipy`, `simpy`, `gurobipy`, `matplotlib`, and `pandas`.
- The current project should migrate reusable logic into `src/` with explicit configuration, modular environment dynamics, and reproducible smoke tests.
- Do not edit legacy files unless intentionally preserving provenance is not important for that file.
