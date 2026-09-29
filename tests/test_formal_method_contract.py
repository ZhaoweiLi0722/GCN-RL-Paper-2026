import unittest

from evaluation.audit_formal_method_contract import ALGORITHMS, EXPECTED, validate_config


class FormalMethodContractTests(unittest.TestCase):
    def fixture(self):
        config = {"algorithm": ALGORITHMS[0], "seed": 10}
        for field, value in EXPECTED.items():
            node = config
            parts = field.split(".")
            for key in parts[:-1]:
                node = node.setdefault(key, {})
            node[parts[-1]] = value
        return config

    def test_valid_contract(self):
        self.assertEqual(validate_config(self.fixture(), ALGORITHMS[0], 10), EXPECTED)

    def test_misidentified_seed_or_algorithm(self):
        for algorithm, seed in [(ALGORITHMS[1], 10), (ALGORITHMS[0], 11)]:
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_config(self.fixture(), algorithm, seed)

    def test_every_explicit_field_is_checked(self):
        for field in EXPECTED:
            with self.subTest(field=field):
                config = self.fixture()
                node = config
                parts = field.split(".")
                for key in parts[:-1]:
                    node = node[key]
                node[parts[-1]] = None
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    validate_config(config, ALGORITHMS[0], 10)

    def test_missing_reward_mode_rejected(self):
        config = self.fixture()
        del config["residual_action"]["online_reward_mode"]
        with self.assertRaisesRegex(ValueError, "missing contract"):
            validate_config(config, ALGORITHMS[0], 10)


if __name__ == "__main__":
    unittest.main()
