import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import io as io
from data import df_train, df_test
from torchvision import transforms



class MNISTDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((32, 32), Image.BICUBIC)
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float).view(32, 32)
        img_tensor /= 255.0
        img_tensor = 1.0 - img_tensor
        img_tensor = img_tensor.unsqueeze(0)
        label = int(row['label'])
        return img_tensor, label


train_transform = transforms.Compose([
    transforms.RandomRotation(15), 
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  
    transforms.RandomHorizontalFlip(),  
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
])

class MNISTDataset2(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  

        img = img.resize((32, 32), Image.BICUBIC)

        if self.transform:
            img = self.transform(img)
        img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float).view(32, 32)
        img_tensor /= 255.0 
        img_tensor = 1.0 - img_tensor  

        img_tensor = img_tensor.unsqueeze(0)

        label = int(row['label'])
        return img_tensor, label