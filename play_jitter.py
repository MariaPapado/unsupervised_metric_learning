import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import numbers

#------- ColorJitter -------#
class ColorJitter(object):
    def __init__(self, color_adjust_para):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
        """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
        '''Ajust brightness, contrast, saturation, hue'''
        '''Input: PIL Image, Output: PIL Image'''
        self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

    def __call__(self, in_img, idx):
        if idx==0:
            print('0')
            brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            print(brightness_factor)
            in_img = F.adjust_brightness(in_img, brightness_factor)

        if idx==1:
            print('1')
            contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            print(contrast_factor)
            in_img = F.adjust_contrast(in_img, contrast_factor)

        if idx==2:
            print('2')
            saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            print(saturation_factor)
            in_img = F.adjust_saturation(in_img, saturation_factor)

        if idx==3:
            print('3')
            hue_factor = np.random.uniform(-self.hue, self.hue)
            print(hue_factor)
            in_img = F.adjust_hue(in_img, hue_factor)

        #in_img = np.asarray(in_img)
        #in_img = in_img.clip(0,1)

        return in_img
    
im = Image.open('/home/mariapap/CODE/pickle_dataset/train/im2/grtgazparis_59.png')
#im = np.array(im)

color_adjust_para = [0.2, 0.2, 0.2, 0.2]
Img1_Jitter = ColorJitter(color_adjust_para)

im = Img1_Jitter(im, np.random.randint(0,4))
#print(im.shape)
#im = Image.fromarray(im*255)
print('immmmmmmmmm', np.array(im).shape)
im.save('RANDOM.png')

#hue=0.1692522903754216
#hue_factor = 0.1692522903754216 #np.random.uniform(hue, hue)
#im = F.adjust_hue(im, hue_factor)
#im.save('HUE.png')
