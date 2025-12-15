from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import math
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import Conv2d
import torch.nn.functional as F

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim, bias = False),
            nn.GELU(),
            #nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim, bias = False),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim, bias = False),
            #nn.LayerNorm(emb_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class PatchEmbeddingLinear(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, dim):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.linear = nn.Linear(patch_size * patch_size*in_chans, dim, bias=False)

    def forward(self, x):
        # x: (B, 1, H, W)
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size)\
                   .unfold(3, self.patch_size, self.patch_size)  # (B, 1, num_patches_h, num_patches_w, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, C * self.patch_size * self.patch_size)  # (B, num_patches, C*patch_size*patch_size)

        out = self.linear(patches)  # (B, num_patches, dim)
        return out

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim, bias = False),
        #    nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,dim,  bias = False),
            
        )
    def forward(self,x):
        x=self.net(x)
        return x

class PerPixelFC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias =False)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.fc(x)              # [B, H, W, out_channels]
        x = x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]
        return x

class MixerBlock(nn.Module):
    def __init__(self,dim,num_patch,token_dim,channel_dim,dropout=0.):
        super().__init__()
        self.token_mixer=nn.Sequential(
        #    nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch,token_dim,dropout),
            Rearrange('b d n -> b n d')
 
         )
        self.channel_mixer=nn.Sequential(
        #    nn.LayerNorm(dim),
            FeedForward(dim,channel_dim,dropout)
        )
        #self.layer_normal = nn.BatchNorm1d(num_patch)

    def forward(self,x):
        x = x+self.token_mixer(x)
        x = x+self.channel_mixer(x)
        #x = self.layer_normal(x)
        return x
    
class MLPMixer(nn.Module):
    def __init__(self,in_channels,dim,patch_size,num_patches,depth,token_dim,channel_dim,dropout=0.):
        super().__init__()

        self.num_patches = num_patches
        self.dim = dim
        self.depth = depth

        self.to_embedding = nn.Sequential(
                                            PatchEmbeddingLinear(4, patch_size,4, dim),
                                             )

        
        self.mixer_blocks=nn.ModuleList([])
        self.temb_blocks = nn.ModuleList([])
        
        self.temb_blocks = EmbedFC(1, self.dim)

        for _ in range(self.depth-1):
            self.mixer_blocks.append(MixerBlock(dim,self.num_patches,token_dim,channel_dim,dropout))
        #    self.temb_blocks.append(EmbedFC(1, self.dim))
        
        self.conv_out = nn.Sequential(
                        PerPixelFC(dim, dim),
                        PerPixelFC(dim, 4)
                      )
    def forward(self, x, t):       
        
        lattice = x.shape[-1] 
        x = self.to_embedding(x)
        #temb = self.temb_blocks(t)
        #x = x + temb.reshape(-1,1, self.dim)
        for i, mixer_block in enumerate(self.mixer_blocks):
            temb = self.temb_blocks(t)
            x = x + temb.reshape(-1,1, self.dim)
            x = mixer_block(x) 
        
        x = x.transpose(1, 2)  
         
        x = x.view(-1,self.dim,lattice,lattice)

        x = self.conv_out(x)
        return x
        


