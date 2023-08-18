import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones.vision_transformer import Attention, PreNorm, FeedForward, FeedForwardShrink


class DoubleConv(nn.Module):
    '''(conv - bn - relu) * 2'''
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        ) 
            
    def forward(self, X):
        X = self.doubleconv(X)
        return X


class Up(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.doubleconv = DoubleConv(in_channels, out_channels, mid_channels)
        
    def forward(self, X1, X2):
        X1 = self.up(X1)
        diffY = torch.tensor([X2.size()[2] - X1.size()[2]])
        diffX = torch.tensor([X2.size()[3] - X1.size()[3]])
        # just incase:
        X1 = F.pad(X1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        X = torch.cat([X2, X1], dim=1)
        X = self.doubleconv(X)
        return X

class AttnUp(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_heads=8):
        super(AttnUp, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.doubleconv = DoubleConv(in_channels, out_channels, mid_channels)

        self.attn_layers = nn.ModuleList([])
        self.attn_layers.append(nn.ModuleList([
            PreNorm(in_channels, Attention(in_channels, num_heads=num_heads, return_x_only=True)),
            PreNorm(in_channels, FeedForward(in_channels, in_channels, dropout=0.))]
            ))
        
    def forward(self, X1, X2):
        X1 = self.up(X1)
        diffY = torch.tensor([X2.size()[2] - X1.size()[2]])
        diffX = torch.tensor([X2.size()[3] - X1.size()[3]])
        # just incase:
        X1 = F.pad(X1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        X = torch.cat([X2, X1], dim=1)

        b, c, h, w = X.shape
        X = X.permute(0, 2, 3, 1).view(b, h*w, c)
        for attn, ff in self.attn_layers:
            X = attn(X) + X
            X = ff(X) + X
        X = X.view(b, h, w, c).permute(0, 3, 1, 2)


        X = self.doubleconv(X)
        return X