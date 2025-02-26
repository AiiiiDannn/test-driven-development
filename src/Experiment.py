import numpy as np
import matplotlib.pyplot as plt
from SignalDetection import SignalDetection


class Experiment:
    def __init__(self):
        self.conditions = []

    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        if not isinstance(sdt_obj, SignalDetection):
            raise TypeError("Input must be a SignalDetection object.")
        self.conditions.append((sdt_obj, label))


    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        if not self.conditions:
            raise ValueError("No conditions added to the experiment")

        roc_points = []
        for sdt_obj, _ in self.conditions:
            roc_points.append((sdt_obj.false_alarm_rate(), sdt_obj.hit_rate()))

        # Both lists are sorted in ascending order of false alarm rate (first element).
        roc_points.sort(key=lambda x: x[0])

        false_alarm_rates, hit_rates = zip(*roc_points)    # Hinted by ChatGPT: Python’s zip(*iterable) transposes a list of tuples, effectively splitting them into separate groups. It’s a concise way to separate paired values.
        return list(false_alarm_rates), list(hit_rates)

    def compute_auc(self) -> float:
        if not self.conditions:
            raise ValueError("No conditions added to the experiment")

        false_alarm_rates, hit_rates = self.sorted_roc_points()

        auc_value = 0
        for i in range(len(false_alarm_rates) - 1):
            width = false_alarm_rates[i + 1] - false_alarm_rates[i]
            height = (hit_rates[i + 1] + hit_rates[i]) / 2
            auc_value += width * height

        return auc_value

    def plot_roc_curve(self, show_plot: bool = True):
        if not self.conditions:
            raise ValueError("No conditions added to the experiment")

        false_alarm_rates, hit_rates = self.sorted_roc_points()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(false_alarm_rates, hit_rates, marker="o", linestyle="-", label="ROC Curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance Level")
        ax.set_xlabel("False Alarm Rate")
        ax.set_ylabel("Hit Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True)

        if show_plot:
            plt.show()

        return fig


if __name__ == "__main__":
    sdt1 = SignalDetection(10, 20, 15, 5)    # FA = 0.75, HR = 0.33
    sdt2 = SignalDetection(5, 5, 10, 5)    # FA = 0.67, HR = 0.5
    exp = Experiment()
    exp.add_condition(sdt1, "Condition A")
    exp.add_condition(sdt2, "Condition B")
    print(exp.sorted_roc_points())
    print(exp.compute_auc())
