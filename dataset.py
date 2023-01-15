
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader



# CustomDataset

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        self.data_dir = "./data"

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.img_path_list[index])
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.label_list is not None:
            label = torch.FloatTensor(self.label_list[index])
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)
