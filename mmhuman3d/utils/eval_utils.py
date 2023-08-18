import numpy as np
import torch

class Dice:
    def __init__(self, average='micro', threshold=0.1):
        self.average = average
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, target):
        pred = pred > self.threshold
        target = target.astype(bool)
        self.tp += (pred & target).sum()
        self.fp += (pred & ~target).sum()
        self.tn += (~pred & ~target).sum()
        self.fn += (~pred & target).sum()

    def compute(self):
        if self.average == 'micro':
            return 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        elif self.average == 'macro':
            return (2 * self.tp / (2 * self.tp + self.fp + self.fn)).mean()
        else:
            raise ValueError(f'Unsupported average type {self.average}')

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

class IOU:
    def __init__(self, average='micro', threshold=0.1):
        self.average = average
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, target):
        pred = pred > self.threshold
        target = target.astype(bool)
        self.tp += (pred & target).sum()
        self.fp += (pred & ~target).sum()
        self.tn += (~pred & ~target).sum()
        self.fn += (~pred & target).sum()

    def compute(self):
        if self.average == 'micro':
            return self.tp / (self.tp + self.fp + self.fn)
        elif self.average == 'macro':
            return (self.tp / (self.tp + self.fp + self.fn)).mean()
        else:
            raise ValueError(f'Unsupported average type {self.average}')

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
