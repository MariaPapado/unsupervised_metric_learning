import torch
import torch.nn as nn

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"


def MahalanobisDis(x, xhat):
    # This function computes the Mahalanobis Distance between  x and xhat
    # assume shape(x) = shape(xhat) = b, c, h, w
    [b, c, h, w] = x.shape
    
    # Reshaping x and xhat into 1D vectors
    x = x.view(b, c, h*w)
    xhat = xhat.view(b, c, h*w)

    #Computing covariance matrix
    x_mean = torch.mean(x, dim=2, keepdim=True)
    xhat_mean = torch.mean(xhat, dim=2, keepdim=True)
    S = torch.bmm((x-x_mean), (xhat-xhat_mean).permute(0, 2, 1))
    print('SSSSSSSSSSSSSS', S.shape)
    S = torch.linalg.inv(S)
    

#    print('aaaaa', torch.bmm((x-x_mean).permute(0, 2, 1)).shape)
    #Computing MahalanobisDis
    print('0', (x-x_mean).permute(0, 2, 1).shape)
    print('1', S.shape)
    print('2', (xhat-xhat_mean).shape)
    DM = torch.mean(torch.bmm(torch.bmm((x-x_mean).permute(0, 2, 1), S), (xhat-xhat_mean)))
    print('DM', DM)
    return DM


p=128
a = torch.rand(1,2,p,p).cuda()
b = torch.rand(1,2,p,p).cuda()

print('a', a)
print('b', b)

out = MahalanobisDis(a,b)
print(out)


part_losses = []

for x in range(0, p, p/2):

  for y in range(0, p, p/2):
    prob_part_a = a[:,:,






