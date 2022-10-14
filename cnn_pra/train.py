import torch
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision
from dataset.poc import CustomImageDataset
from torchvision.models.segmentation import fcn_resnet50
from lossfunc.lossfunc import LossFunc


lr = 0.001
epochs = 10000
batch_size = 64
max_endure = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = CustomImageDataset("F:/CNN_pra/data/poc/dataset.csv")
train_dl = DataLoader(train_ds, batch_size=batch_size)
model = fcn_resnet50(num_classes=2).to(device)
optim = torch.optim.SGD(model.parameters(), lr)
lossfunc = LossFunc()
model.train()

hist = []
best_loss = None
best_statedict = None
for epoch in tqdm(range(epochs)):
    
    for feature, truth in train_dl:
        time_sta = time.time()
        feature = feature.to(device)
        truth = truth.to(device)
        pred = model(feature)
        loss = lossfunc(pred, truth)
        print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        time_end = time.time()
        print(time_end - time_sta)
        

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

    hist.index()

    hist.append(loss)

torch.save(best_statedict, "../data/statedict/statedict.pt")
