import torch 
from torch import nn 
import torch.nn.functional as F 
from attention import SelfAttention 

class VAE_ResidualBlock(nn.Module):
    def __init__(self , in_channels , out_channels): 
        super().__init__()

        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels ,out_channels , kernel_size=3 , padding=1)

        self.group_norm2 = nn.GroupNorm(32, in_channels)
        self.conv2 = nn.Conv2d(in_channels ,out_channels , kernel_size=3 , padding=1)

        if in_channels == out_channels : 
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_channels ,out_channels , kernel_size=1, padding=0)
        
    
    def forward(self,x: torch.Tensor ) -> torch.Tensor:
        # x : ( batch_size , in_channels , height ,width)

        # let make a copy of x for residual connection afterwards
        residue = x 

        x = self.group_norm1(x) 
        x = F.silu(x)
        x = self.conv1(x)

        x = self.group_norm2(x) 
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residual_layer(residue)
    


class VAE_AttentionBlock(nn.Module):
    def __init__(self , channels : int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1,channels)

    def forward(self , x:torch.Tensor) -> torch.Tensor:

        # x : ( batch_size , channels , height , width)
        residue = x 

        n ,c ,h ,w = x.shape
        # ( batch_size , channels , height , width)--> ( batch_size , channels , height*width)
        x = x.view(n, c ,h*w)

        # ( batch_size , channels , height , width)--> ( batch_size ,height*width, channels)
        x = x .transpose(-1,-2)

        # ( batch_size , channels , height , width)--> ( batch_size ,height*width, channels)
        x = self.attention(x)
        
        # ( batch_size ,height*width, channels )--> ( batch_size , channels ,height*width)
        x = x .transpose(-1,-2)

        #  ( batch_size , channels , height*width)--> ( batch_size , channels , height , width)
        x = x.view(n,c,h,w)

        x += residue

        return x 