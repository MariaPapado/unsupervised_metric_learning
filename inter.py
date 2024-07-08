import torch

a = torch.rand(4,4)

print(a)

a = torch.nn.functional.interpolate(a.unsqueeze(0).unsqueeze(0), scale_factor=1/2**1, mode='nearest')
print(a)
print(a.shape)
