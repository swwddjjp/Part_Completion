import os
import torch
import torch.nn as nn

path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))+'/'
print(path)
import sys
sys.path.append(path+'distance/pyTorchChamferDistance/chamfer_distance/')
from chamfer_distance import ChamferDistance
sys.path.append(path+'distance/PyTorchEMD/')
from emd import earth_mover_distance

class L2_ChamferLoss(nn.Module):
    def __init__(self):
        super(L2_ChamferLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist


class L2_ChamferEval(nn.Module):
    def __init__(self):
        super(L2_ChamferEval, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(dist1) + torch.mean(dist2)
        return dist * 10000


class L1_ChamferLoss(nn.Module):
    def __init__(self):
        super(L1_ChamferLoss, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        # print(dist1, dist1.shape) [B, N]
        dist = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
        return dist / 2


class L1_ChamferEval(nn.Module):
    def __init__(self):
        super(L1_ChamferEval, self).__init__()
        self.chamfer_dist = ChamferDistance()

    def forward(self, array1, array2):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        dist = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
        return dist / 2 * 1000


class F1Score(nn.Module):
    def __init__(self):
        super(F1Score, self).__init__()
        self.chamfer_dist = ChamferDistance()
    
    def forward(self, array1, array2, threshold=0.0001):
        dist1, dist2 = self.chamfer_dist(array1, array2)
        precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
        precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
        fscore[torch.isnan(fscore)] = 0
        return fscore, precision_1, precision_2


class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, array1, array2):
        dist = earth_mover_distance(array1, array2, transpose=False)
        # print(dist.shape)
        dist = torch.mean(dist) / array1.shape[1]
        return dist
        

class EMDEval(nn.Module):
    def __init__(self):
        super(EMDEval, self).__init__()

    def forward(self, array1, array2):
        dist = earth_mover_distance(array1, array2, transpose=False)
        dist = torch.mean(dist) / array1.shape[1]
        return dist * 100
