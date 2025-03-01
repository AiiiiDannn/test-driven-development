import unittest
from SignalDetection import SignalDetection
from Experiment import Experiment
import numpy as np
import matplotlib.pyplot as plt


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
        with self.assertRaises(TypeError):
            self.exp.add_condition("invalid_object", "Invalid")


    # Test sorted_roc_points

    def test_sorted_roc_points(self):
        sdt1 = SignalDetection(10, 0, 5, 10)    # FA = 0.33, HR = 1.0
        sdt2 = SignalDetection(5, 5, 10, 5)     # FA = 0.67, HR = 0.5
        sdt3 = SignalDetection(8, 2, 3, 12)     # FA = 0.2, HR = 0.8

        self.exp.add_condition(sdt1, "A")
        self.exp.add_condition(sdt2, "B")
        self.exp.add_condition(sdt3, "C")

        fa_rates, h_rates = self.exp.sorted_roc_points()
        self.assertEqual(fa_rates, sorted(fa_rates))

        # Ensure hit rates are correctly paired with the FA rates 
        expected_pairs = sorted([(sdt.false_alarm_rate(), sdt.hit_rate()) for sdt, _ in self.exp.conditions])    # Sort by FA rate, suggested by ChatGPT
        actual_pairs = list(zip(*self.exp.sorted_roc_points()))
        self.assertEqual(actual_pairs, expected_pairs)
    
    def test_sorted_roc_points_with_empty_conditions(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()

    
    # Test compute_auc

    def test_compute_auc_known_cases(self):
        sdt1 = SignalDetection(0, 10, 0, 10)
        sdt2 = SignalDetection(10, 0, 10, 0)
        self.exp.add_condition(sdt1, "(0,0)")
        self.exp.add_condition(sdt2, "(1,1)")
        self.assertAlmostEqual(self.exp.compute_auc(), 0.5, places=3)

    def test_compute_auc_perfect_case(self):
        sdt1 = SignalDetection(0, 10, 0, 10)
        sdt2 = SignalDetection(10, 0, 0, 10)
        sdt3 = SignalDetection(10, 0, 10, 0)
        self.exp.add_condition(sdt1, "(0,0)")
        self.exp.add_condition(sdt2, "(0,1)")
        self.exp.add_condition(sdt3, "(1,1)")
        self.assertAlmostEqual(self.exp.compute_auc(), 1.0, places=3)

    def test_compute_auc_general_case(self):
        sdt1 = SignalDetection(3, 7, 4, 6)
        sdt2 = SignalDetection(10, 2, 6, 8)
        sdt3 = SignalDetection(18, 2, 12, 8) 
        sdt4 = SignalDetection(25, 5, 20, 10) 
        
        self.exp.add_condition(sdt1, "A")
        self.exp.add_condition(sdt2, "B")
        self.exp.add_condition(sdt3, "C")
        self.exp.add_condition(sdt4, "D")

        # Using numpy to compute AUC for comparison
        fa_rates, h_rates = self.exp.sorted_roc_points()
        manual_auc = self.exp.compute_auc()
        numpy_auc = np.trapezoid(h_rates, fa_rates)

        self.assertAlmostEqual(manual_auc, numpy_auc, places=3)


    def test_compute_auc_with_empty_conditions(self):
        with self.assertRaises(ValueError):
            self.exp.compute_auc()


    # Test plot_roc_curve
    def test_plot_roc_curve(self):
        sdt1 = SignalDetection(10, 0, 5, 10)
        sdt2 = SignalDetection(5, 5, 10, 5)
        
        self.exp.add_condition(sdt1, "Condition A")
        self.exp.add_condition(sdt2, "Condition B")
        
        plt_obj = self.exp.plot_roc_curve(show_plot=False)

        self.assertIsInstance(plt_obj, plt.Figure)

        ax = plt_obj.gca()
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)  # Should have 2 lines (ROC curve + diagonal)

        self.assertEqual(ax.get_xlabel(), "False Alarm Rate")
        self.assertEqual(ax.get_ylabel(), "Hit Rate")
        self.assertEqual(ax.get_title(), "ROC Curve")

    def test_plot_roc_curve_with_empty_conditions(self):
        with self.assertRaises(ValueError):
            self.exp.plot_roc_curve(show_plot=False)





if __name__ == "__main__":
    unittest.main()
