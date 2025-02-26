import unittest
from SignalDetection import SignalDetection
from Experiment import Experiment
from sklearn.metrics import auc    # https://stackoverflow.com/questions/66397641/which-is-the-correct-way-to-calculate-auc-with-scikit-learn


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.exp = Experiment()


    # Test init

    def test_init(self):
        self.assertIsInstance(self.exp, Experiment)
        self.assertEqual(len(self.exp.conditions), 0)

    # Test add_condition
    def test_add_valid_condition(self):
        sdt = SignalDetection(40, 10, 10, 40)
        self.exp.add_condition(sdt, "Condition A")
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertIsInstance(self.exp.conditions[0][0], SignalDetection)
        self.assertEqual(self.exp.conditions[0][1], "Condition A")
    
    def test_add_multiple_conditions(self):
        sdt1 = SignalDetection(10, 5, 5, 10)
        sdt2 = SignalDetection(15, 10, 10, 15)
        self.exp.add_condition(sdt1, "Condition 1")
        self.exp.add_condition(sdt2, "Condition 2")
        self.assertEqual(len(self.exp.conditions), 2)
        self.assertIsInstance(self.exp.conditions[0][0], SignalDetection)
        self.assertIsInstance(self.exp.conditions[1][0], SignalDetection)
        self.assertEqual(self.exp.conditions[0][1], "Condition 1")
        self.assertEqual(self.exp.conditions[1][1], "Condition 2")

    def test_condition_without_label(self):
        sdt = SignalDetection(10, 5, 5, 10)
        self.exp.add_condition(sdt)
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertIsInstance(self.exp.conditions[0][0], SignalDetection)
        self.assertIsNone(self.exp.conditions[0][1])
    
    def test_add_invalid_condition(self):
        with self.assertRaises(ValueError):
            self.exp.add_condition("invalid_object", "Invalid")


    # Test sorted_roc_points

    def test_sorted_roc_points(self):
        sdt1 = SignalDetection(10, 0, 5, 10)
        sdt2 = SignalDetection(5, 5, 10, 5)
        sdt3 = SignalDetection(8, 2, 3, 12)

        self.exp.add_condition(sdt3, "C")
        self.exp.add_condition(sdt1, "A")
        self.exp.add_condition(sdt2, "B")

        fa_rates, h_rates = self.exp.sorted_roc_points()
        self.assertEqual(fa_rates, sorted(fa_rates))

        # Ensure hit rates are correctly paired with the FA rates
        expected_pairs = list(zip(fa_rates, h_rates))
        actual_pairs = list(zip(*self.exp.sorted_roc_points()))
        self.assertEqual(actual_pairs, expected_pairs)
    
    def test_sorted_roc_points_with_empty_conditions(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()

    
    # Test compute_auc

    def test_compute_auc_known_cases(self):
        sdt1 = SignalDetection(0, 0, 0, 0)
        sdt2 = SignalDetection(10, 0, 10, 0)
        self.exp.add_condition(sdt1, "(0,0)")
        self.exp.add_condition(sdt2, "(1,1)")
        self.assertAlmostEqual(self.exp.compute_auc(), 0.5, places=3)

    def test_compute_auc_perfect_case(self):
        sdt1 = SignalDetection(0, 0, 0, 0)
        sdt2 = SignalDetection(10, 0, 0, 10)
        sdt3 = SignalDetection(10, 0, 10, 0)
        self.exp.add_condition(sdt1, "(0,0)")
        self.exp.add_condition(sdt2, "(0,1)")
        self.exp.add_condition(sdt3, "(1,1)")
        self.assertAlmostEqual(self.exp.compute_auc(), 1.0, places=3)

    def test_compute_auc_with_empty_conditions(self):
        with self.assertRaises(ValueError):
            self.exp.compute_auc()





if __name__ == "__main__":
    unittest.main()
