import torch
from torch import nn


class LossFunc(nn.Module):
    h = 250
    w = 100

    def __init__(self):
        super(LossFunc, self).__init__()
        self.__lossfunc = torch.nn.CrossEntropyLoss()

    def forward(self, p, t: torch.Tensor):
        p: torch.Tensor = p["out"]
        loss = 0
        for h in range(self.h):
            for w in range(self.w):
                loss += self.__lossfunc(p[:, :, h, w], t[:, 0, h, w])
        return loss
