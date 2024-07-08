""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))

        self.inc_img = (DoubleConv(n_channels, 64))
        self.down1_img = (Down(64, 128))
        self.down2_img = (Down(128, 256))
        self.down3_img = (Down(256, 512))

        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.down4_img = (Down(512, 1024 // factor))

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.sigm = torch.nn.Softmax()

    def forward(self, img1, img2):
        feats_img1, feats_img2 = [], []

        x = torch.abs(img1-img2)

        #f1_1, f1_2 = self.inc(img1), self.inc(img2)
        #x1 = torch.abs(f1_1 - f1_2)
        #f2_1, f2_2 = self.down1(f1_1), self.down1(f1_2)
        #x2 = torch.abs(f2_1-f2_2)
        #f3_1, f3_2 = self.down2(f2_1), self.down2(f2_2)
        #x3 = torch.abs(f3_1-f3_2)
        #f4_1, f4_2 = self.down3(f3_1), self.down3(f3_2)
        #x4 = torch.abs(f4_1-f4_2)
        #f5_1, f5_2 = self.down4(f4_1), self.down4(f4_2)
        #x5 = torch.abs(f5_1-f5_2)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1_img1 = self.inc_img(img1)
        x2_img1 = self.down1_img(x1_img1)
        x3_img1 = self.down2_img(x2_img1)
        x4_img1 = self.down3_img(x3_img1)
        x5_img1 = self.down4_img(x4_img1)

        feats_img1.append(x1_img1)
        feats_img1.append(x2_img1)
        feats_img1.append(x3_img1)
        feats_img1.append(x4_img1)
        feats_img1.append(x5_img1)

        x1_img2 = self.inc_img(img2)
        x2_img2 = self.down1_img(x1_img2)
        x3_img2 = self.down2_img(x2_img2)
        x4_img2 = self.down3_img(x3_img2)
        x5_img2 = self.down4_img(x4_img2)

        feats_img2.append(x1_img2)
        feats_img2.append(x2_img2)
        feats_img2.append(x3_img2)
        feats_img2.append(x4_img2)
        feats_img2.append(x5_img2)

        #print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigm(logits), feats_img1, feats_img2