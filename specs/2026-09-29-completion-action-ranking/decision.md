# Decision after S3: continuation adequacy, not a DDPG launch

Status: proposal only, 2026-09-29. No follow-up computation is authorized or
executed by this document. S3's finite implementation/run/audit/readout chain
is complete. E1 domain inputs remain missing; Stage E remains closed.

## Recommended decision

Do not enlarge S3 samples, train DDPG/TD3, retune costs or claim that a frozen
policy is near-optimal. The useful finding is narrower: the old low-budget
rollout choices overstate the case for a sophisticated controller, and the
remaining apparent advantage over a public feedback rule is not replicated.

The next defensible diagnostic, **only with new scope approval**, is a bounded
comparison of feedback continuations on the same archived synthetic contexts.
Its question is whether the current post-first-action rule changes the ranking
enough to invalidate the one-request rollout screen. It is not a search for a
scenario with a desired RL sign.

## Proposed comparison design

- Retain all four unique public contexts and six aliases, the same action grid,
  costs, booked jobs, delayed actuation and full-settlement accounting. No new
  patient scenario, actual-world rollout, hidden-world information or reward
  change. Hold model-response assumptions fixed to separate continuation error
  from model error; do not silently add belief updating to only one comparator.
- Include the existing booked rule and the already implemented reservation
  rule. Each can use public pending hours, support queues and bookings; neither
  may access future completion uniforms or the private family label.
- Add one genuinely event-conditioned feedback lookahead, not a fixed two-
  request prefix. Its branch actions must depend only on observations already
  available at that branch. The one-interval action lead must be retained.
  Define a fixed, small lookahead and terminal reference before execution;
  quantify its numerical error before calling it a credible comparator.
- Use paired full-settlement costs, fresh independent selection/validation
  blocks, frozen choices and explicitly budgeted model-query counts. Publish
  results for every continuation/context, including ties and worse costs.
  Never select a continuation using the eventual validation results.
- Freeze a separate protocol, exact branch/forecast budgets, one-run compute
  ceiling and failure/closure rules before any recorded matrix. No claim of
  exact optimality or a strong MPC bound follows merely from adding lookahead.
  If a credible comparator cannot fit the declared cap, report that limitation
  instead of automatically relaxing it.

The scope request is to authorize this finite diagnostic packet, including its
prospective protocol, implementation, tests, one bounded model-only run and
audit. It does **not** authorize subsequent training or actual-world evaluation.
The exact executable budget must be committed before observing new outcomes.

## Interpretation gates

| Possible result | Permissible interpretation and next action |
| --- | --- |
| Rankings depend materially on the fallback | S2/S3 are controller-approximation diagnostics, not model-independent headroom evidence. Stop short of neural training. |
| A stronger feedback continuation reproducibly improves conditional cost | Feedback may have value under the assumed model. Next seek separately approved, matched-information out-of-sample actual-world validation and domain support, not an automatic online-DDPG claim. |
| A simple feedback rule matches the bounded comparator, or advantages remain unstable | Do not promote this synthetic channel to a neural campaign on the present evidence. Retain the negative result and focus the manuscript on supported contributions. |
| Closure, numerical accuracy or evidence integrity fails | Preserve failure evidence; no automatic repair/retry or enlarged budget. |

Even in the favorable row, a future online-learning claim must compare the same
frozen and online-updated initialization with identical available information,
plus adaptive heuristics and system-identification/control baselines. Distinguish
improvement due to feedback, inference, architecture, offline preparation and
online parameter updates. Report compute/sample costs as well as objective cost.

## Publication implications

The paper can gain credibility from a precise boundary and reproducible negative
attribution analysis; this is not a guarantee of acceptance or enough novelty by
itself. Do not relabel model integration precision as scientific replication.
Do not weaken the baseline to make online RL look useful. New clinical relevance
requires the still-missing staffing/response/cost evidence, not another synthetic
success story. Existing formal comparisons and their methodological limitations
must remain visible independently of this exploratory appendix.

Concrete approval question: may we open the bounded **model-only feedback-
continuation comparison** above, while keeping training, new patient/actual-world
episodes, reward tuning, formal data and remote operations closed?
