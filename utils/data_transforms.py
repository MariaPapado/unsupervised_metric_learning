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

    def __call__(self, in_imgs, idx, iter):
#        if self.brightness > 0:

        b_size = in_imgs.shape[0]
        batch = []

        for b in range(0, b_size):
            in_img = in_imgs[b]
            if idx==0:
                #brightness_factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                brightness_factor = np.random.uniform(0.5,1.4)
                in_img = F.adjust_brightness(in_img, brightness_factor)

    #        if self.contrast > 0:
            if idx==1:
                #contrast_factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                contrast_factor = np.random.uniform(0.5,1.4)
                in_img = F.adjust_contrast(in_img, contrast_factor)

    #        if self.saturation > 0:
            if idx==2:
                #saturation_factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                saturation_factor = np.random.uniform(0.5,1.4)
                in_img = F.adjust_saturation(in_img, saturation_factor)

    #        if self.hue > 0:
#            if idx==3:
#                #hue_factor = np.random.uniform(-self.hue, self.hue)
#                hue_factor = np.random.uniform(-0.5,0.5) 
#                in_img = F.adjust_hue(in_img*255, hue_factor)
            #in_img = in_img/255.0
            #print('unique', np.unique(in_img.data.cpu().numpy()))
            #img_save = in_img.permute(1,2,0).data.cpu().data.numpy()
            #cv2.imwrite('./chk/im_{}_{}.png'.format(idx,iter), img_save*255)
            batch.append(in_img.unsqueeze(0))


        f_batch = torch.cat(batch, 0)

        #in_img = np.asarray(in_img)
        #in_img = in_img.clip(0,1)
        
        return f_batch