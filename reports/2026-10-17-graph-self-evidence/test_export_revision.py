"""Pure arithmetic checks; no research runtime, model, or trajectory calls."""

import unittest

from export_revision import reverse_display


class ReverseDisplayTests(unittest.TestCase):
    def test_denominator_and_interval_order(self):
        pair = dict(candidate="a", reference="b", saving_fraction=0.2,
                    saving_fraction_ci95=[0.1, 0.3], extra_patient_losses=4,
                    extra_patient_losses_ci95=[-2, 8])
        reverse = reverse_display(pair)
        self.assertEqual((reverse["candidate"], reverse["reference"]), ("b", "a"))
        self.assertAlmostEqual(reverse["saving_fraction"], -0.25)
        self.assertAlmostEqual(reverse["saving_fraction_ci95"][0], -0.3 / 0.7)
        self.assertAlmostEqual(reverse["saving_fraction_ci95"][1], -0.1 / 0.9)
        self.assertEqual(reverse["extra_patient_losses_ci95"], [-8, 2])

    def test_round_trip_including_negative_gain(self):
        pair = dict(candidate="a", reference="b", saving_fraction=-0.1,
                    saving_fraction_ci95=[-0.2, 0.05], extra_patient_losses=-3,
                    extra_patient_losses_ci95=[-8, 2])
        twice = reverse_display(reverse_display(pair))
        for key in ("saving_fraction", "extra_patient_losses"):
            self.assertAlmostEqual(twice[key], pair[key])
        for key in ("saving_fraction_ci95", "extra_patient_losses_ci95"):
            for actual, expected in zip(twice[key], pair[key]):
                self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
