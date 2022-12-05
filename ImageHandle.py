from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import os
from Contant import DEVICE
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, img_dir, file_names, transform=None):
        self.file_names = file_names
        self.img_dir = img_dir
        if transform == None:
            self.transform = torch.nn.Sequential(transforms.CenterCrop((300, 300)), transforms.Resize((400, 400)))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        label = filename.split('.')[0]
        res_img = image.to(DEVICE)
        return res_img, label

def get_img_dataloader(dir, filenames, n_batch):
    return DataLoader(ImageDataset(dir, filenames), n_batch, shuffle=True)