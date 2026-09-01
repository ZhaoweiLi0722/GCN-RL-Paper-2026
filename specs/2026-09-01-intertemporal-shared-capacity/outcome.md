# Locked Development Outcome

Date: 2026-09-01

Decision: **close this configuration at J2**. J3, J4, DDPG training, and formal
confirmation are not authorized.

## Evidence chain

### J1 mechanics

The 20-clinic patient environment completed a 12-step, four-policy smoke with
the same final RNG digest in every arm. Requests respected the `31.5` shared
budget, matured after the declared lead time, and remained finite. This was an
engineering check, not a scientific comparison.

### J2a episode pre-screen

Discovery selected static budget fraction `0.25` and forecast/graph budget
fraction `1.0`. On 24 fresh validation pairs across three schedule variants:

- forecast versus tuned static: `+0.7410%` pooled mean, 17/24 positive;
- graph forecast versus tuned static: `+0.7787%` pooled mean, 18/24 positive;
- graph forecast had positive mean saving in every variant and was clinically
  noninferior in aggregate.

J2a therefore authorized the fixed-state J2 screen, but not training.

### J2 fixed-state screen

The prospective screen used 54 fixed states, 10 legal allocation candidates,
two discovery worlds and three fresh validation worlds per state: 2,700 paired
rows in total. All 1,080 local perturbations changed the continuous action
interface and all 270 paired worlds had matching end-of-stream RNG hashes.

The strongest pooled discovery comparator was `b1.00_a0.25`. Frozen
state-specific discovery choices produced only `+0.06915%` cost saving on
validation (`+0.06930% +/- 0.02794%`, normal 95% half-width across nine
scenario-seed clusters). The result was clinically noninferior in aggregate,
57.4% of selections were interior, and 54 realized allocation vectors were
distinct. However:

- prospective value was below the required `0.5%`;
- zero of 54 states achieved a clinically noninferior `0.5%` saving;
- the J2 decision was `fixed_state_j2_failed`.

### Post-hoc diagnostic

A clearly labeled post-hoc upper bound selected the best candidate separately
for every state using the validation outcomes themselves. Even this optimistic
oracle saved only `0.09356%`; its best state saved `0.30084%`, and zero states
reached `0.5%`. Discovery choices agreed with the validation oracle in 50% of
states.

The failure is therefore primarily candidate-library saturation, with some
label instability, rather than an integer execution failure. Training DDPG on
this closed configuration would be scientifically unjustified.

## Immutable provenance

- J2a rows SHA256:
  `9326e6155b4216962ea113fca5640ba1d4cc110a832f10b03d72158caeb68f34`
- J2a summary SHA256:
  `97bd9c65a4bb20167b36414c494e366ada5086ca643a62179a90c45a333b803d`
- J2 config SHA256:
  `c1e16497ac68f7a246c4bced0bf1f2be35bc3a6aeed0472d89603cf2290b1ae2`
- J2 rows SHA256:
  `ad488114a2a30c21acf53bdcf2eb59ff6864aa2ccab9775d5168cc77cb6bfcb4`
- J2 summary SHA256:
  `3af63a13e770e12926f2f726e6ffc34585878ca85e89f529c285aca1bd2bbff4`
- post-hoc diagnostic SHA256:
  `119e7cc2f7c68df773a76458634cb9ef4eca219964a4b7a88cb52d76474ebaa6`

Any future study must use a new specification, seed families, and result root.
It must test materially different per-facility residual allocations before any
learning, not retune this failed grid or weaken its gates.
