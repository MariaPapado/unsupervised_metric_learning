from __future__ import print_function
import matplotlib.pyplot as plt
%matplotlib inline

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

#Importing different models
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip

import torch
import torch.optim
import random
#random.seed(42)
from utils.common_utils import *
from utils.cd_utils import *
from utils.encoder_features import VGG_Net, Encoder_Net
from torchvision import transforms as F
from utils.data_transforms import ColorJitter

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score


from utils.loss import info_loss, feature_InfoNce_loss, cm_InfoNce_loss

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64
print('ok')


# Testing image number in OSCD dataset.
IMAGE_N    = 7 #Beirut(22), Lasvegas(7), cupertino(18)
# Testing image list in OSCD dataset.
IMAGE_LIST = ['brasilia','montpellier','norcia','rio','saclay_w','valencia','dubai','lasvegas','milano','chongqing', 'aguasclaras','bercy','bordeaux','nantes','paris','rennes','saclay_e','abudhabi','cupertino','pisa','beihai','hongkong','beirut','mumbai']
# Set path for pre-change, post-change images.
# CM is used only to calculate the final metrics. We dont use it for training.
img1_path    = '/home/mariapap/Music/METRIC_unsupervised_test/'
img2_path    = '/home/mariapap/Music/METRIC_unsupervised_test/'
cm_path      = '/home/mariapap/Music/METRIC_unsupervised_test/'
print("Change detection on |>>>>> " + IMAGE_LIST[IMAGE_N] + " <<<<<| in OSCD dataset")



# Notation
# img1 = pre-change image
# img2 = post-change image
img1, img2 = get_onera_RGBimg(img1_path, img2_path)
# Printing shape of the original MSI images
# bands, W, H = img2.shape


# Notation: cm = binary change map
cm = get_onera_cm(cm_path) 
if cm.ndim == 3:
    cm = cm[:, :, 1]


# Converting numpy arrays to pytorch tensors
img1_var = np_to_torch(img1).type(dtype)
img2_var = np_to_torch(img2).type(dtype)
mask_var = np_to_torch(cm).type(dtype)
#----- Color Jittering parameters -----#
color_adjust_para = [0.2, 0.2, 0.2, 0.2]
#Img1 and Img2 color jittering
Img1_Jitter = ColorJitter(color_adjust_para)
Img2_Jitter = ColorJitter(color_adjust_para)
img1_j      = Img1_Jitter(img1_var.cpu())[0]
img2_j      = Img2_Jitter(img2_var.cpu())[0]
#----- Visualizing Color Jittered Example Image -----#
img1j_rgb = (img1_j-np.amin(img1_j))/(np.amax(img1_j)-np.amin(img1_j))
img2j_rgb = (img2_j-np.amin(img2_j))/(np.amax(img2_j)-np.amin(img2_j))
cm        = cm/np.amax(cm)

###############################################################################################################################################
# General parameters
pad           = 'reflection' # 'zero'
OPT_OVER      = 'net'
OPTIMIZER     = 'adam'
net_input     = torch.abs(img1_var-img2_var)
input_depth   = 3
show_every    = 5
figsize       = 12
reg_noise_std = 0.03

# Parameters of Deep Change Dector
LR               = 0.0001
reg_noise_std    = 0.03
num_iter         = 50
threshold        = 0.7
N_vgg            = 3
requires__grad   = True

distance_measure  = torch.nn.L1Loss()
distance_measure2 = torch.nn.L1Loss()
InfoNCE_loss      = info_loss(temp_c = 1, temp_nc = 1)

# Define change prediction network
net = ResNet(input_depth, 2, 32, 32, need_sigmoid=True, act_fun='LeakyReLU')

#Experimenting with other network architectures
#net = skip(input_depth, 2, 
           #num_channels_down = [128] * 5,
           #num_channels_up =   [128] * 5,
           #num_channels_skip =    [128] * 5,  
           #filter_size_up = 3, filter_size_down = 3, 
           #upsample_mode='nearest', filter_skip_size=1,
           #need_softmax=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
         
net = net.type(dtype)

Encoder= VGG_Net(resize=False, requires__grad=requires__grad).cuda()

#Experimenting with different transfer learning methods.
#Encoder           = Encoder_Net().cuda()
#pretrained_dict   = torch.load("/media/lidan/ssd/Chaminda/change_detection/pretrain_encoder/outputs/best_model.pth")
#Encoder_dict      = Encoder.state_dict()
#pretrained_dict   = {k: v for k, v in pretrained_dict.items() if k in Encoder_dict}
#Encoder_dict.update(pretrained_dict) 
#Encoder.load_state_dict(Encoder_dict)

Encoder.eval()
for param in Encoder.parameters():
    param.requires_grad = requires__grad


#Printing number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)
###############################################################################################################################################



i           = 0
AUC_best    = 0.0
change_best = np.empty(cm.shape)
def closure():
    
    global i
    global AUC_best
    global change_best
    
    net_input = net_input_saved
    
    # Getting change and unchange map
    out = net(net_input)
    
    # Change probability (Pc) and unchanged probability (Pc') map
    pred_cm   = out[0, 0, :, :]
    pred_ncm  = 1.0 - pred_cm
    
    # Image features
    img1_feat = Encoder(np_to_torch(img1).type(dtype))
    img2_feat = Encoder(np_to_torch(img2).type(dtype))
    
    # -------- pixel to pixel loss function -------#
    # Applying color jittering
    img1_aug = Img1_Jitter(img1_var.cpu())[0]
    img2_aug = Img2_Jitter(img2_var.cpu())[0]

    # Feature level similarity and dissimilarity from VGG pretrained
    img1_aug_feat = Encoder(np_to_torch(img1_aug).type(dtype))
    img2_aug_feat = Encoder(np_to_torch(img2_aug).type(dtype))
    
    #--------- Loss FUNCTIONS-------
    [loss_s_img, loss_d_img]           = image_loss(img1_var, img2_var, pred_cm, pred_ncm, distance_measure, distance_measure)
    [loss_s_feat, loss_d_feat]         = feature_loss(img1_feat, img2_feat, pred_cm, pred_ncm, distance_measure, distance_measure, N_vgg)
    [loss_s_feat_aug, loss_d_feat_aug] = feature_loss(img1_aug_feat, img1_aug_feat, pred_cm, pred_ncm, distance_measure, distance_measure, N_vgg)
    loss_aug                           = contrastive_loss(img1_feat, img1_aug_feat, distance_measure2, 3) + contrastive_loss(img2_feat, img2_aug_feat, distance_measure2, 3)
    loss_cm                            = cm_InfoNce_loss(pred_cm, pred_ncm, InfoNCE_loss)

    total_loss  = (loss_s_img + loss_d_img) + (loss_s_feat + loss_d_feat) + (loss_s_feat_aug + loss_d_feat_aug) + loss_aug + loss_cm
            
    total_loss.backward()
    
    # Ploting predicted change map and ground thruth
    print ('Iteration %05d    Loss %f'  % (i, total_loss.item()), '\r', end='')
    if  PLOT and i % show_every == 0:
        change_np   = np.expand_dims(torch_to_np(pred_cm.detach().unsqueeze(0)), 0)
        nochange_np = np.expand_dims(torch_to_np(pred_ncm.detach().unsqueeze(0)), 0)
        
        target_np = np.expand_dims(cm, 0)
        
        plot_image_grid([change_np, nochange_np, target_np], factor=10, nrow=3)
        
        #Printing precision, recall, F1 score
        change_np_binary = (change_np>threshold).astype(int).flatten()
        target_np_binary = target_np.astype(int).flatten()
        precision        = precision_score(target_np_binary, change_np_binary)
        recall           = recall_score(target_np_binary, change_np_binary)
        f1               = f1_score(target_np_binary, change_np_binary)
        fpr, tpr, thres = roc_curve(target_np_binary, change_np.flatten())
        AUC = auc(fpr, tpr)
        print('Precision: %.3f  Recall: %.3f    F1-Score: %.3f     AUC: %.3f' % (precision, recall, f1, AUC))
        
        if AUC>AUC_best:
            AUC_best = AUC
            change_best = change_np
        
    i += 1
    return total_loss

net_input_saved = net_input.detach().clone()
noise           = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

##########################################################################################################################