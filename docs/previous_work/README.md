# Previous Work References

This folder stores small reference artifacts from the previous capacity-planning project.

- `DynRes.pdf`
- `GNN-RL-Knowledge-based.docx`
- `tseng_2025_deep_rl_dynamic_capacity_planning_decentralised_prm.pdf`

Keep these files as background material. Current experiment code, configs, and reproducible outputs should live under `src/`, `experiments/`, and ignored result folders rather than being mixed into this reference folder.

## Tseng et al. IJPR Setting Takeaway

The IJPR DRL capacity-planning paper models a decentralised PRM manufacturing
network, not a single centralized manufacturing site. Each manufacturing
facility has local bioreactors/reagents and receives patient specimens; the
control problem includes reagent replenishment, inventory transshipment,
capacity relocation, and demand/specimen sharing. Their numerical study uses a
two-facility decentralised network with weekly decision epochs, where
transshipments at epoch `t` arrive before epoch `t+1`.

For our 20-clinic geography scenario, that weekly transshipment assumption is a
useful prior-work abstraction, but the realistic B&B/Wiley transfer data are in
hours. We therefore keep the main patient-condition geography scenario as a
weekly decision model with continuous hour-level transfer-time metadata/costs,
rather than forcing every transfer into a one-week delay.
