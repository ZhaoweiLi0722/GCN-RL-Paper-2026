"""SAC baseline placeholder.

SAC is intentionally left as a second-phase implementation so the flat DDPG and
TD3 baselines can be validated first. Do not report SAC results until this file
contains a real entropy-regularized off-policy implementation and smoke tests.
"""


class SACAgent:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SAC is planned but not implemented in this baseline pass.")
