import torch
from torch.utils.data import DataLoader
from dataset.poc import CustomImageDataset
from torchvision.models.densenet import DenseNet
from loss.lossfunc import LossFunc

batch_size = 16
pred_path = "../50split/pred.csv"
test_path = "../50split/test.csv"
img_dir = "../50split/"
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_ds = CustomImageDataset(test_path, img_dir)
eval_dl = DataLoader(eval_ds, batch_size=batch_size)
model = DenseNet(num_classes=250).to(device)
statedict = torch.load("../data/statedict/statedict50.pt")
model.load_state_dict(statedict)
lossfunc = LossFunc()
model.eval()

with torch.no_grad():
    with open(pred_path, mode="w") as f:
        for feature, truth in eval_dl:
            feature = feature.to(device)
            truth = truth.to(device)
            batch_pred = model(feature)
            print(batch_pred)

            pred: torch.Tensor
            for pred in batch_pred:
                f.write(",".join(map(str, pred)) + "\n")
