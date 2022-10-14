import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision
from dataset.poc import CustomImageDataset
from torchvision.models.segmentation import fcn_resnet50
from lossfunc.lossfunc import LossFunc


batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_ds = CustomImageDataset("F:/CNN_pra/data/poc/test.csv")
eval_dl = DataLoader(eval_ds, batch_size=batch_size)
model = fcn_resnet50(num_classes=2).to(device)
statedict = torch.load("../data/statedict/statedict.pt")
model.load_state_dict(statedict)
lossfunc = LossFunc()
model.eval()

hist = []
with torch.no_grad():
    for feature, truth in eval_dl:
        feature = feature.to(device)
        truth = truth.to(device)
        batch_pred = model(feature)

        pred: torch.Tensor
        for pred in batch_pred:
            pred = pred.argmax(0)
            pred = (pred*255).to(torch.uint8)
            image = Image.fromarray(pred.to("cpu").detach().numpy().copy())
            image.save("hoge.jpg", quality=100)
