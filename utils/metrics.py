import pandas as pd
import numpy as np

#create average meter
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


#compute binary accuracy
def binary_accuracy(output, target):
    pred = output.cpu() >= 0.5
    accuracy = (pred.int().eq(target.int())).sum()
    accuracy = accuracy * 100 / np.prod(np.array(target.size()))
    return accuracy



