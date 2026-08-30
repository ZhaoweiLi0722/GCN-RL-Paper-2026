# Routing Label Budget Study (Step-0 Draft)

> **STATUS: DRAFT — NOT AUTHORIZED.** This spec proposes re-examining one
> input to a closed result and therefore goes through change control before
> anything runs. It does not amend the routing-primary campaign, reopen the
> formal holdout, or authorize any training. Its only possible outcomes are
> (a) the Stage G1 label-instability conclusion is confirmed as fundamental,
> or (b) it is reclassified as underpowered — which would *reopen a question*,
> not change any manuscript claim by itself.

## The question

Stage G1 closed the online-DDPG extension on the finding that routing's
counterfactual best-action labels do not replicate: discovery and validation
streams agreed on the best legal action in only **54.5%** of states against a
70% gate. That conclusion feeds the manuscript's online-attribution negative.

Two facts learned since make its evidentiary basis worth checking:

1. **G1's labels were estimated from very few worlds** — 3 discovery and 5
   validation replications per action (`patient_indexed_specimen_routing_ddpg_legal_action_ranker_g1.json`,
   `dataset.discovery_replications` / `validation_replications`).
2. **Best-action agreement in this simulator is budget-sensitive.** On the
   overtime channel's 12,960 rows, agreement between disjoint world groups
   rises 0.826 → 0.852 → 0.876 → 0.887 as worlds-per-estimate go 1 → 2 → 3 → 4.
   And CRN pairing here is *exact* (verified: identical RNG end-state,
   enrollment counts, and patient ids across arms), so disagreement between
   groups is genuine world-to-world variation in the best action — which more
   worlds legitimately average away. The operations-research literature
   reports that common random numbers plus sequential-halving allocation
   inside the labelling step cuts optimality gaps by an order of magnitude at
   fixed budget (Temizöz et al., EJOR 2025, DOI 10.1016/j.ejor.2025.01.026).

So: **was 54.5% a property of the routing channel, or of an 8-world budget?**

## The design in one paragraph

Collect a fixed pool of paired-CRN rollout outcomes — every legal specimen
action evaluated on every world, for a fixed set of anchor-policy decision
states — then answer every budget question *offline* by replaying allocation
policies against the pool. One simulation campaign, no adaptive contamination,
and both uniform and sequential-halving allocation can be scored on identical
data. The primary output is an **agreement-versus-budget curve** with a
prospective reading rule fixed in this spec.

- [Protocol](protocol.md) — states, actions, budgets, estimators, reading
  rules, and pre-registered prediction
- [Validation](validation.md) — tests and guards

## Sign-off record

| Role | Name | Decision | Date |
| --- | --- | --- | --- |
| Principal investigator | Howard Tseng | _pending_ | |
| Collaborator / reviewer | Zhaowei Li | _pending — hands-off for this branch; to review at final debrief_ | |
