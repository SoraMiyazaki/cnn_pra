import time
import os

import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.densenet import DenseNet

from dataset.poc import CustomImageDataset
from loss.lossfunc import LossFunc

lr = 0.1
epochs = 2000
batch_size = 64
max_endure = 10
# datapath = "../data/poc/bk_dataset.csv"
datapath = "../split/dataset.csv"
# img_dir = "../data/poc/"
img_dir = "../split/"
device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = CustomImageDataset(datapath, img_dir)
train_dl = DataLoader(train_ds, batch_size=batch_size)
test_path = "../split/test.csv"
img_dir = "../split/"
eval_ds = CustomImageDataset(test_path, img_dir)
eval_dl = DataLoader(eval_ds, batch_size=batch_size)
model = DenseNet(num_classes=250).to(device)
optim = torch.optim.SGD(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
lossfunc = LossFunc()

hist_train = []
hist_test = []
best_loss = None
best_statedict = None

def get_test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for feature_test, truth_test in eval_dl:
            feature_test = feature_test.to(device)
            truth_test = truth_test.to(device)
            pred_test = model(feature_test)
            test_loss += float(lossfunc(pred_test, truth_test).item())
        test_loss /= len(eval_dl) 
    return test_loss

for epoch in tqdm(range(epochs)):
    model.train()
    for feature, truth in tqdm(train_dl):    
        feature = feature.to(device)
        truth = truth.to(device)
        pred = model(feature)
        loss_train = lossfunc(pred, truth)
        optim.zero_grad()
        loss_train.backward()
        optim.step()

    scheduler.step()

    loss_train = float(loss_train)
    if not best_loss:
        best_loss = loss_train
        best_statedict = model.state_dict()
        endure = 0
    elif loss_train < best_loss:
        best_loss = loss_train
        best_statedict = model.state_dict()
        endure = 0
    else:
        endure += 1

    if endure > max_endure:
        break

    # hist.index()

    hist_train.append(loss_train)
    hist_test.append(get_test())

torch.save(best_statedict, "../data/statedict/statedict50.pt")

np.savetxt("hist_train.csv", hist_train, fmt='%e')
np.savetxt("hist_test.csv", hist_test, fmt='%e')