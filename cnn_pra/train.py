import time

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.densenet import DenseNet

from dataset.poc import CustomImageDataset
from loss.lossfunc import LossFunc

lr = 0.001
epochs = 1000
batch_size = 32
max_endure = 10
datapath = "../data/poc/dataset.csv"
img_dir = "../data/poc/"
device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = CustomImageDataset(datapath, img_dir)
train_dl = DataLoader(train_ds, batch_size=batch_size)
model = DenseNet(num_classes=250).to(device)
optim = torch.optim.SGD(model.parameters(), lr)
lossfunc = LossFunc()

model.train()
hist = []
best_loss = None
best_statedict = None

for epoch in tqdm(range(epochs)):
    
    for feature, truth in tqdm(train_dl):
        # time_sta = time.time()

        feature = feature.to(device)
        truth = truth.to(device)
        pred = model(feature)

        # time_end = time.time()
        # print(time_end - time_sta)
        # time_sta = time.time()

        loss = lossfunc(pred, truth)

        # time_end = time.time()
        # print(time_end - time_sta)
        # time_sta = time.time()

        optim.zero_grad()

        # time_end = time.time()
        # print(time_end - time_sta)
        # time_sta = time.time()

        loss.backward()

        # time_end = time.time()
        # print(time_end - time_sta)
        # time_sta = time.time()

        optim.step()

        # time_end = time.time()
        # print(time_end - time_sta)

        

    loss = float(loss)
    if not best_loss:
        best_loss = loss
        best_statedict = model.state_dict()
        endure = 0
    elif loss < best_loss:
        best_loss = loss
        best_statedict = model.state_dict()
        endure = 0
    else:
        endure += 1

    if endure > max_endure:
        break

    # hist.index()

    hist.append(loss)

torch.save(best_statedict, "../data/statedict/statedict.pt")
