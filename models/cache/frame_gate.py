"""
Frame-level coarse screening.

TODO:
Replace simple difference metric with learned metric if necessary.
"""

import torch


class FrameGate:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def __call__(self, current, previous):
        diff = torch.mean(torch.abs(current - previous))
        return diff

    def need_full_compute(self, current, previous):
        score = self(current, previous)
        return score > self.threshold
