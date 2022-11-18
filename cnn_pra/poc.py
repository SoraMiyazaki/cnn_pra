import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

data_num = 1000

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir: str):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        # self.img_labels = self.img_labels.sample(n=data_num, random_state=0)
        self.transform = transforms.ToTensor()
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        feature_path = self.img_labels.iloc[idx, 0]
        feature_path = os.path.join(self.img_dir, feature_path)
        feature = self.transform(Image.open(feature_path).convert("RGB"))

        truth = self.img_labels.iloc[idx, 1]

        return feature, truth
