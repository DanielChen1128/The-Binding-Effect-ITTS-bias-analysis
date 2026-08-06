import unittest

import pandas as pd

import analyze_gender


class GenderStatisticsTests(unittest.TestCase):
    def test_paper_and_adult_denominators_are_both_reported(self):
        analyze_gender.pd = pd
        frame = pd.DataFrame({
            "predicted_gender": ["female", "male", "child", "unknown"],
        })
        overall = analyze_gender.compute_statistics(frame)["overall"]
        self.assertEqual(overall["total"], 3)
        self.assertEqual(overall["adult_total"], 2)
        self.assertAlmostEqual(overall["female_probability"], 1 / 3)
        self.assertAlmostEqual(overall["adult_female_probability"], 1 / 2)
        self.assertEqual(overall["excluded"], {"unknown": 1})


if __name__ == "__main__":
    unittest.main()
