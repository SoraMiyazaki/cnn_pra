import torch
from torch import nn


class LossFunc(nn.Module):
    h = 250
    w = 100

    def __init__(self):
        super(LossFunc, self).__init__()
        self.__lossfunc = torch.nn.CrossEntropyLoss()

    def forward(self, p_batch, t_batch: torch.Tensor):
        return self.__lossfunc(p_batch, t_batch)

