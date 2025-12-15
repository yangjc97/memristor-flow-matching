import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim, bias = False),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim, bias = False),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim, bias = False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            #nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
            #nn.BatchNorm2d(out_channels),
            #nn.GELU(),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.conv1(x)
            return x1

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ ResidualConvBlock(in_channels, out_channels, is_res = True), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels,if_fill = False):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias = False),
            #ResidualConvBlock(out_channels, out_channels, is_res = True),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, n_feat = 16, kernel_size = 3, lattice_shape = 14):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.kernel_size = kernel_size
        self.if_fill = (lattice_shape[0] % 4 != 0)

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        

        self.down1 = UnetDown(n_feat, 1 * n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        self.up0 = ResidualConvBlock(2*n_feat, 2*n_feat, is_res=True)

        
        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        self.up1 = UnetUp(4 * n_feat, n_feat, self.if_fill)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        self.out = nn.Sequential(
            nn.Conv2d(2*  n_feat, n_feat, kernel_size, 1, 1, bias = False),
            nn.BatchNorm2d(n_feat), 
            nn.ReLU(),

            nn.Conv2d(n_feat, self.in_channels , kernel_size, 1, 1, bias = False),
        )

    def forward(self, x, t):

        x = self.init_conv(x)
        
        
        down1 = self.down1(x)

        down2 = self.down2(down1)

        up1 = self.up0(down2) 
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)

        up2 = self.up1(up1 + temb1, down2)  # add and multiply embeddings
         
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        up3 = self.up2(up2 + temb2, down1)

        out = self.out(torch.cat((up3, x), 1))
        return out

