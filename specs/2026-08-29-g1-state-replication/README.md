# Stage G1 State Replication (Step-0 Draft)

> **STATUS: DRAFT — NOT AUTHORIZED. REQUIRES THE RTX 4090 HOST.**
> This is the only study that can adjudicate whether Stage G1's label
> instability was fundamental or budget-limited, because it is the only design
> that uses G1's own states. It cannot run on Howard's laptop: the 156
> frozen-pretrain trajectory states depend on Stage F1 checkpoint artifacts
> that live on the 4090 host.

## Why this exists

Stage G1 closed the online-DDPG extension on 54.5% best-action agreement
against a 70% gate, from only 3 discovery and 5 validation worlds per action.
The routing label budget study (2026-08-29) tried to test whether that was a
budget artefact and **could not**: at G1's exact budget shape it measured
0.821 agreement on fresh MDL-2-anchored states, against G1's 0.545. It never
reproduced G1's baseline, so it could not test what lifts that baseline.

The gap is therefore attributable to **states and policy context**, not
replication budget — but that leaves the original question open. If G1's
labels do stabilise at higher budget on *its own* states, then the finding
that closed the extension is a budget statement, and the Stage F1/G0/G1 chain
that the manuscript's online-attribution negative rests on would need
revisiting.

That is a consequential enough possibility to be worth one careful study, and
narrow enough to be worth exactly one.

## The design in one paragraph

Reconstruct G1's 156 frozen-pretrain states from the immutable Stage F1
training trees, enumerate the same five legal specimen actions, and evaluate
every action on a pool of 64 worlds per state instead of 3+5. Then replay
budget policies offline against the pool, exactly as the laptop study did.
**The study is gated on first reproducing G1's own number**: if the pool's
3+5-shaped agreement does not land near 0.545, the replication has failed and
the study reports itself uninformative rather than drawing a conclusion.

That precondition gate is the direct lesson of the laptop study, where its
absence let a mechanically-correct rule return a scientifically wrong
classification.

- [Protocol](protocol.md)

## Sign-off record

| Role | Name | Decision | Date |
| --- | --- | --- | --- |
| Principal investigator | Howard Tseng | _pending_ | |
| Collaborator / reviewer | Zhaowei Li | _pending — owns the 4090 host_ | |
