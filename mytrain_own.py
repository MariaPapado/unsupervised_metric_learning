from custom_data import ChangeDetection
import argparse
import transform as tr
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchnet as tnt
from tqdm import tqdm
import shutil
#import tools
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

import unet_parts
import unet_model

#from __future__ import print_function
#import matplotlib.pyplot as plt
#%matplotlib inline

#import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#Importing different models
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import unet

import torch.optim
#random.seed(42)
from utils.common_utils import *
from utils.cd_utils import *
from utils.encoder_features import VGG_Net, Encoder_Net
from torchvision import transforms as F
from utils.data_transforms import ColorJitter
import cv2
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
from pytorch_metric_learning import losses

from utils.loss import info_loss, feature_InfoNce_loss, cm_InfoNce_loss

import torch.nn.functional as F

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/home/mariapap/CODE/pickle_dataset/")
parser.add_argument("--train_txt_file", type=str, default="/home/mariapap/CODE/pickle_dataset/train/im2_hist_reg")
parser.add_argument("--val_txt_file", type=str, default="/home/mariapap/CODE/pickle_dataset/val/im1")
#parser.add_argument("--data_root", type=str, default="/home/mariapap/DATASETS/LEVIR-CD-256/")
#parser.add_argument("--train_txt_file", type=str, default="/home/mariapap/DATASETS/LEVIR-CD-256/list/train.txt")
#parser.add_argument("--val_txt_file", type=str, default="/home/mariapap/DATASETS/LEVIR-CD-256/list/val.txt")
parser.add_argument("--batch-size", type=int, default=3)
parser.add_argument("--val-batch-size", type=int, default=1)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#with open(args.train_txt_file) as file:
#    train_ids = [line.rstrip() for line in file]
train_ids = os.listdir(args.train_txt_file)
print('lll', len(train_ids))

##########################################
val_lines = train_ids[::15]
print(len(val_lines))
train_ids = list(set(train_ids) - set(val_lines))
print(len(train_ids))
print(len(val_lines))


###############################################

#train_ids = train_ids[:800]
trainset = ChangeDetection(args.data_root, 'train', train_ids)


#with open(args.val_txt_file) as file:
#    val_lines = [line.rstrip() for line in file]
#val_lines = os.listdir(args.val_txt_file)
#val_lines = val_lines[:200]
valset = ChangeDetection(args.data_root, 'train', val_lines)

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=False, drop_last=True)
valloader = DataLoader(valset, batch_size=args.val_batch_size, shuffle=True,
                                    pin_memory=True, drop_last=True)


print(len(trainloader))
print(len(valloader))


net = unet_model.UNet(3,2)
#net = ResNet(3, 2, 32, 32, need_sigmoid=True, act_fun='LeakyReLU')
net.cuda()
#print(net)
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)
epochs = 50


color_adjust_para = [0.2, 0.2, 0.2, 0.2]
Img_Jitter = ColorJitter(color_adjust_para)

distance_measure  = torch.nn.L1Loss()
distance_measure2 = torch.nn.L1Loss()
distance_measure3 = torch.nn.L1Loss()
InfoNCE_loss      = info_loss(temp_c = 1, temp_nc = 1)

total_iters = len(trainloader) * epochs
print('totaliters', total_iters)
save_folder = 'saved_models' #where to save the models and training progress
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.mkdir(save_folder)

vis_folder = 'VAL_VIS' #where to save the models and training progress
if os.path.exists(vis_folder):
    shutil.rmtree(vis_folder)
os.mkdir(vis_folder)

ff=open('./' + save_folder + '/progress.txt','w')
N_vgg=2
num_classes = 2
iter_ = 0
for epoch in range(1, epochs+1):

    #tbar = tqdm(trainloader)
    net.train()
    #conf = np.zeros((num_classes, num_classes))


    for i, batch in enumerate(tqdm(trainloader)):
        img1, img2 = batch
        img1, img2 = img1.float().to(device), img2.float().to(device)

        #rand_idx = np.random.randint(0,4)
        img1_aug = Img_Jitter(img1, np.random.randint(0,3), i)
        img2_aug = Img_Jitter(img2, np.random.randint(0,3), i)
        #print(img1_aug.shape, img2_aug.shape)
        #print(imgs.shape)

        optimizer.zero_grad()
   
        out, img1_feat, img2_feat, img1_aug_feats, img2_aug_feats = net(img1, img2, img1_aug, img2_aug)  
        #print('shapes', img1_feat[0].shape)
        #print('shapes', len(img1_aug_feats)) 
        #print('shapes', len(img2_feat))
        #print('shapes', len(img2_aug_feats)) 
        #out = net(img1, img2)    
        #print('out', out.shape)    
        # Change probability (Pc) and unchanged probability (Pc') map

        pred_cm   = out[:, 1, :, :]
        pred_ncm  = 1.0 - pred_cm

        loss_s_img, loss_d_img = image_loss(img1, img2, pred_cm, pred_ncm, distance_measure, distance_measure2)
        loss_s_feat, loss_d_feat = feature_loss(img1_feat, img2_feat, pred_cm, pred_ncm, distance_measure, distance_measure2, N_vgg)
        closs_s_feat, closs_d_feat = contrastive_loss(img1_feat, img1_aug_feats, distance_measure3, N_vgg), contrastive_loss(img2_feat, img2_aug_feats, distance_measure3, N_vgg)
#        closs_s_feat, closs_d_feat = distance_measure3(img1, img1_aug), distance_measure3(img2, img2_aug)

        #loss_s_feat_aug, loss_d_feat_aug = feature_loss(img1_feat, img1_aug_feats, pred_cm, pred_ncm, distance_measure, distance_measure2, N_vgg), feature_loss(img2_feat, img2_aug_feats, pred_cm, pred_ncm, distance_measure, distance_measure2, N_vgg)

        #print('ffff', loss_s_feat, loss_d_feat)
        #print('aaaaa', loss_s_feat_aug, loss_d_feat_aug)

        #loss_feat_aug = loss_s_feat_aug + loss_d_feat_aug
        loss_img = loss_s_img + loss_d_img
        loss_feat = loss_s_feat + loss_d_feat
        loss_c = closs_s_feat + closs_d_feat
        loss_cm = cm_InfoNce_loss(pred_cm, pred_ncm, InfoNCE_loss)

        total_loss = loss_img + loss_feat + loss_c + loss_cm 

        total_loss.backward()

        optimizer.step()
        iter_ += 1
        #lr_ = base_lr * (1.0 - iter_ / total_iters) ** 0.9
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr_


        if iter_ % 20 == 0: #print loss for IMG, FEAT, CM, TOTAL
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_IMG_D: {:.3f}\tLoss_IMG_S: {:.3f}\tLoss_FEAT_S: {:.3f}\tLoss_FEAT_D: {:.3f}\tLoss_CFEAT_S: {:.3f}\tLoss_CFEAT_D: {:.3f}\tLoss_CM: {:.3f}\tLoss_TOTAL: {:.3f}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss_d_img.item(), loss_s_img.item(), loss_s_feat.item(), loss_d_feat.item(), closs_s_feat.item(), closs_d_feat.item(), loss_cm.item(), total_loss.item()))



    with torch.no_grad():
        net.eval()
        os.mkdir('./{}/epoch{}'.format(vis_folder, str(epoch)))
        for i, batch in enumerate(tqdm(valloader)):
            img1, img2 = batch
            img1, img2 = img1.float().to(device), img2.float().to(device)
            img1_aug = Img_Jitter(img1, np.random.randint(0,4), i)
            img2_aug = Img_Jitter(img2, np.random.randint(0,4), i)

            out, img1_feat, img2_feat, img1_aug_feats, img2_aug_feats = net(img1, img2, img1_aug, img2_aug)    
            #out = net(img1, img2)        

            # Change probability (Pc) and unchanged probability (Pc') map
            pred_cm   = out[0, 1, :, :]
            #print('cm', pred_cm.shape)
            img1_s = img1.squeeze().permute(1,2,0).data.cpu().numpy()*255
            img2_s = img2.squeeze().permute(1,2,0).data.cpu().numpy()*255
            #out_s = F.softmax(out, 1)
            out_s = torch.argmax(out,1)
            out_s = out_s.squeeze().data.cpu().numpy()*255
            #print('unique', np.unique(out_s))
            #print('SSSSSSSSSSSSSSSSSSSSSSSSS', img1_s.shape, img2_s.shape, out_s.shape)

            if i%30==0:
                os.mkdir('./{}/epoch{}/iter_{}/'.format(vis_folder, str(epoch), str(i)))
                cv2.imwrite('./{}/epoch{}/iter_{}/img1_{}.png'.format(vis_folder, str(epoch), str(i), str(i)), img1_s)
                cv2.imwrite('./{}/epoch{}/iter_{}/img2_{}.png'.format(vis_folder, str(epoch), str(i), str(i)), img2_s)
                cv2.imwrite('./{}/epoch{}/iter_{}/pred_{}.png'.format(vis_folder, str(epoch), str(i), str(i)), out_s)            
                cv2.imwrite('./{}/epoch{}/iter_{}/prob0_{}.png'.format(vis_folder, str(epoch), str(i), str(i)), out.squeeze()[0].data.cpu().numpy()*255)            
                cv2.imwrite('./{}/epoch{}/iter_{}/prob1_{}.png'.format(vis_folder, str(epoch), str(i), str(i)), out.squeeze()[1].data.cpu().numpy()*255)            







#    tools.write_results(ff, save_folder, epoch, train_acc, test_acc, change_acc, non_ch, np.mean(train_losses), np.mean(val_losses), conf)

#    #save model in every epoch
#    torch.save(net.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))







