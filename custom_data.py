import transform as tr
import cv2
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ChangeDetection(Dataset):

    def __init__(self, root, mode, data_ids):
        super(ChangeDetection, self).__init__()
        self.root = root
        self.data_ids = data_ids
        self.mode = mode

        self.ids = data_ids
        random.shuffle(self.ids)

        self.transform = transforms.Compose([
            tr.RandomFlipOrRotate(),
            #tr.RandomHorizontalFlip(),
            #tr.RandomVerticalFlip(),
            #tr.RandomFixRotate(),
            #tr.GBlur(),
            #tr.Sharp(),
            #tr.Contrast(),
            #tr.Nothing(),

        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.449, 0.451, 0.461), (0.191, 0.183, 0.191))
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = Image.open(os.path.join(self.root, self.mode, 'im1', id)) #.convert('RGB')
        img2 = Image.open(os.path.join(self.root, self.mode, 'im2_hist', id)) #.convert('RGB')
        #img1 = Image.open(os.path.join(self.root, 'A', id)) #.convert('RGB')
        #img2 = Image.open(os.path.join(self.root, 'B', id)) #.convert('RGB')
        #change = Image.open(os.path.join(self.root, 'change_noveg', id))


        #img1, img2 = np.array(img1), np.array(img2)
        #img1 = cv2.resize(img1, (128,128), cv2.INTER_NEAREST)
        #img2 = cv2.resize(img2, (128,128), cv2.INTER_NEAREST)

        #if self.mode == 'train':
        #    sample = self.transform({'img1': img1, 'img2': img2})
        #    img1, img2 = sample['img1'], sample['img2']

        img1, img2 = np.array(img1), np.array(img2)
        img1, img2 = self.normalize(img1/255.0), self.normalize(img2/255.0)
#        img1, img2 = self.normalize(img1), self.normalize(img2)

        #img1, img2 = np.transpose(img1, (2,0,1)), np.transpose(img2, (2,0,1))
        #img = torch.cat((img1,img2), 0)

        #change = torch.from_numpy(change)

        return img1, img2

    def __len__(self):
        return len(self.ids)
