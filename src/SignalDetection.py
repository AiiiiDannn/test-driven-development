import scipy.stats as stats  # Recommended from ChatGPT, spicy.stats provides faster computational speed, lower complexity and more accurate results.
import numpy as np


class SignalDetection:
    def __init__(self, hits, misses, falseAlarm, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarm = falseAlarm
        self.correctRejections = correctRejections

    def hit_rate(self):
        if (self.hits + self.misses) == 0:
            return 0
        return self.hits / (self.hits + self.misses)

    def false_alarm_rate(self):
        if (self.falseAlarm + self.correctRejections) == 0:
            return 0
        return self.falseAlarm / (self.falseAlarm + self.correctRejections)

    def d_prime(self):
        hateRate = self.hit_rate()
        falseAlarmRate = self.false_alarm_rate()

        return stats.norm.ppf(hateRate) - stats.norm.ppf(falseAlarmRate)

    def criterion(self):
        hateRate = self.hit_rate()
        falseAlarmRate = self.false_alarm_rate()

        return -0.5 * (stats.norm.ppf(hateRate) + stats.norm.ppf(falseAlarmRate))


if __name__ == "__main__":
    sd = SignalDetection(
        hits=10, misses=20, falseAlarm=15, correctRejections=5
    )  # Just an exmaple valuees, match with my manual calculation.
    print("Hit rate:", sd.hit_rate())
    print("False alarm rate:", sd.false_alarm_rate())

    print("d' value:", sd.d_prime())
    print("Criterion value:", sd.criterion())

    sdt1 = SignalDetection(10, 0, 5, 10)
    print("Hit rate:", sdt1.hit_rate())
    print("False alarm rate:", sdt1.false_alarm_rate())
    print("d' value:", sdt1.d_prime())
    print("Criterion value:", sdt1.criterion())
