import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        feature_path = self.img_labels.iloc[idx, 0]
        feature = self.transform(Image.open(feature_path).convert("RGB"))
        truth_path = self.img_labels.iloc[idx, 1]
        truth = self.transform(Image.open(truth_path)).to(int)
        return feature, truth
