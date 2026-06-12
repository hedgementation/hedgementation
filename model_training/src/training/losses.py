import os
import random
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
load_dotenv()
import torch.nn.functional as F

class IgnoreIntervalLoss(torch.nn.Module):
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        super().__init__()
    
    def forward(self, input, target):
        mask = (input < 0.25) | (input > 0.75)
        input = input & mask
        target = target & mask
        return self.cross_entropy(input, target)
        
class WeightedKLDivLoss(torch.nn.Module):
    def __init__(self, weights=None):
        self.kldiv = torch.nn.KLDivLoss
        pass
    
    def forward(self, input, target):
        pass

class CombinedCrossEntropyMSELoss(torch.nn.Module):
    def __init__(self, weight=None, num_classes=2, reduction="mean", loss_weights=[0.5, 0.5]):
        super().__init__()
        if weight != None:
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.mse = torch.nn.MSELoss(reduction=reduction)
        self.num_classes = num_classes
        #self.register_buffer('loss_weights', torch.tensor(loss_weights, dtype=torch.float32))
    
    def forward(self, input, target):
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float().reshape((-1,self.num_classes,128,128,))
        
        losses = torch.stack([
            self.mse(input, target_one_hot),
            self.cross_entropy(input, target)
        ])
        
        return torch.mean(losses)
    
class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None, num_classes=2, reduction=None):
        super().__init__()
        if weight != None:
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
    
    def forward(self, input, target):
        loss = self.cross_entropy(input, target)
        return loss