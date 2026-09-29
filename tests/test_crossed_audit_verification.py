import unittest

import numpy as np

from evaluation.crossed_design_bootstrap_audit import seed_cluster_bootstrap, two_way_bootstrap, world_cluster_bootstrap
from evaluation.verify_formal_crossed_audit import weighted_two_way


class CrossedVerificationTests(unittest.TestCase):
    def test_constant_difference_has_zero_width(self):
        self.assertEqual(weighted_two_way(np.full((5, 4, 100), -3.0), resamples=50, seed=0), (-3.0, -3.0))

    def test_multiplicity_reproduces_index_resampling(self):
        diff = np.random.default_rng(13).normal(size=(5, 4, 10))
        rng = np.random.default_rng(7)
        kwargs = dict(resamples=300, alpha=.05, rng=rng)
        world_cluster_bootstrap(diff, **kwargs)
        seed_cluster_bootstrap(diff, **kwargs)
        expected = two_way_bootstrap(diff, **kwargs)
        np.testing.assert_allclose(weighted_two_way(diff, resamples=300, seed=7), expected, rtol=0, atol=1e-12)

    def test_shared_worlds_do_not_average_away_across_identical_seeds(self):
        world_values = np.arange(10.0)[None, None, :]
        single = world_cluster_bootstrap(world_values, resamples=300, alpha=.05, rng=np.random.default_rng(4))
        repeated = world_cluster_bootstrap(np.repeat(world_values, 5, axis=0), resamples=300, alpha=.05, rng=np.random.default_rng(4))
        self.assertEqual(single, repeated)
        self.assertGreater(single[1] - single[0], 0)


if __name__ == "__main__":
    unittest.main()
