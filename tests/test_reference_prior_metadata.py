"""P2 against locked R4 metadata and invented raw vectors, never patient steps."""

import unittest

from src.rl.candidate_reference_spec import configuration
from tests import test_candidate_pilot_recovery as base


class PriorMetadataTests(unittest.TestCase):
    def setUp(self):
        base.CandidatePilotRecoveryTests.setUp(self)
        self.cfg = configuration(self.root)

    metadata_shell = base.CandidatePilotRecoveryTests.metadata_shell
    test_full_r4_metadata_schema_counts_and_artificial_raw_inference = (
        base.CandidatePilotRecoveryTests.test_full_r4_metadata_schema_counts_and_artificial_raw_inference)
