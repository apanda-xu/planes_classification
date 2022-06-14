import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image

class Plane(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.samples = []
        self.transform_1 = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize([600,600]),
        ])     
        self.transform_2 = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
        ])
        path = 'data/planes/train.txt' if train else 'data/planes/test.txt'
        with open(path) as f:
            lines = f.readlines()
        for line in tqdm(lines):
            im_path, gt = line.split(' ')
            im = cv2.imread(im_path)
            if im is None:
                im = np.asarray(Image.open(im_path).convert('RGB'))
            im = self.transform_1(im)
            self.samples.append([im, int(gt)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        im, gt = self.samples[index]
        # im = self.transform_2(im)
        gt = torch.Tensor([gt]).long().squeeze(0)
        sample = im, gt
        return sample