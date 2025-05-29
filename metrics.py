import numpy as np
from sklearn.metrics import f1_score

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0 
        self.avg = 0 
        self.sum = 0
        self.count = 0
    
    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def binary_accuracy(output, target):
    pred = output.cpu() >= 0.5
    accuracy = (pred.int().eq(target.int())).sum()
    accuracy = accuracy * 100 / np.prod(np.array(target.size()))
    return accuracy


# def f1_score(output, target):
#     pred = (output.cpu() >= 0.5).int()
#     target = target.cpu().int()

#     tp = (pred * target).sum().item()
#     fp = (pred * (1 - target)).sum().item()
#     fn = ((1 - pred) * target).sum().item()

#     if tp + fp == 0 or tp + fn == 0:
#         return 0.0 

#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)

#     if precision + recall == 0:
#         return 0.0

#     f1 = 2 * (precision * recall) / (precision + recall)

#     return f1 * 100

def f1_score(output, target):
    pred = (output.cpu() >= 0.5).int()
    target = target.cpu().int()

    # Handle both [B] and [B, N] shapes
    if pred.dim() == 1:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

    tp = (pred & target).sum(dim=0).float()
    fp = (pred & (1 - target)).sum(dim=0).float()
    fn = ((1 - pred) & target).sum(dim=0).float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1 * 100