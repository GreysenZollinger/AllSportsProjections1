"""Tests for the current forecasting module (not the retired dashboard code)."""

from __future__ import annotations

import unittest

import sports_forecasting as forecasting


class ProbabilityModelTests(unittest.TestCase):
    def test_equal_ratings_give_even_probability(self) -> None:
        self.assertEqual(forecasting.expected_score(1500, 1500), 0.5)

    def test_higher_rating_has_higher_expected_probability(self) -> None:
        self.assertGreater(forecasting.expected_score(1600, 1500), 0.5)
        self.assertLess(forecasting.expected_score(1400, 1500), 0.5)

    def test_opponent_probabilities_sum_to_one(self) -> None:
        a = forecasting.expected_score(1575, 1490)
        b = forecasting.expected_score(1490, 1575)
        self.assertAlmostEqual(a + b, 1.0, places=12)

    def test_probability_stays_between_zero_and_one(self) -> None:
        probability = forecasting.expected_score(1800, 1200)
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_backtest_handles_empty_history(self) -> None:
        result = forecasting.walk_forward_backtest("NFL", [])
        self.assertEqual(result["metrics"]["games"], 0)
        self.assertIsNone(result["metrics"]["accuracy"])
        self.assertEqual(result["predictions"], [])


if __name__ == "__main__":
    unittest.main()
