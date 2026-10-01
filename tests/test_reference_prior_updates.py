"""Reapply bounded numerical kernel acceptance to the explicit P2 policy type."""

from unittest.mock import patch

from src.rl.candidate_ppo_kernel import CandidatePPOKernel
from tests import test_candidate_ppo_kernel as ppo
from tests import test_candidate_imitation as bc
from tests.test_candidate_policy_rollout import contract
from tests.test_reference_prior_candidate import prior_policy


def prior_kernel(*, mode="online", architecture="graph", message_mode="physical", dtype=None, **changes):
    model = prior_policy(architecture, message_mode)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return CandidatePPOKernel(model, contract(), ppo.settings(**changes), enabled=True, mode=mode,
                              sampling_seed=119, shuffle_seed=121)


class PriorPPOUpdateTests(ppo.CandidatePPOKernelTests):
    def setUp(self):
        guard = patch.object(ppo, "kernel", prior_kernel)
        guard.start()
        self.addCleanup(guard.stop)


class PriorBCUpdateTests(bc.CandidateImitationTests):
    def setUp(self):
        for guard in (patch.object(bc, "policy", prior_policy), patch.object(bc, "kernel", prior_kernel)):
            guard.start()
            self.addCleanup(guard.stop)
