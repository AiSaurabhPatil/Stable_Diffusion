import torch 
from torch import nn 
import torch.nn.functional as F 
from attention import SelfAttention , CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self ,n_embed : int):
        super().__init__()
        self.linear1 = nn.Linear(n_embed , 4*n_embed)
        self.linear2 = nn.Linear(4*n_embed ,4 * n_embed)

    def forward(self,x : torch.Tensor)-> torch.Tensor:
        #x : (1,320)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)

        # x : ( 1, 1280)
        return x 
    
class Unet_AttentionBlock(nn.Module):
    def __init__(self , n_head: int , n_embed: int , context_dim=768):
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32 , channels , eps=1e-6)
        self.conv_input = nn.Conv2d(channels , channels , kernel_size=1 , padding= 0 )

        self.layernorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(n_head ,channels , input_proj_bias=False)

        self.layernorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(n_head , channels , context_dim , input_proj_bias = False)

        self.layernorm3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels , 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4 , channels)

        self.conv_output = nn.Conv2d(channels, channels , kernel_size=1 , padding= 0)

    
    def forward(self,x, context ): 
        # x : (batch_size , channels  ,height, width )
         #context : ( batch_size , seq_len , dim)   dim=768

        residue_long = x 

        x = self.groupnorm(x)

        x = self.conv_input(x)

        n ,c ,h, w = x.shape 

        x = x.view(n, c ,h*w)

        x = x.transpose(-1 ,-2)

        residue_short = x 


        x = self.layernorm1(x)

        x = self.attention1(x)

        x +=residue_short

        residue_short = x 


        x = self.layernorm2(x)

        x = self.attention2(x)

        x += residue_short

        residue_short = x 

        x = self.layernorm3(x) 

        x , gate = self.linear_geglu_1(x).chunk(2,dim = -1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        x = x.transpose(-1 ,-2)

        x = x.view(n, c ,h ,w)

        return self.conv_output(x) + residue_long




class Unet_ResidualBlock(nn.Module):
    def __init__(self , in_channels : int , out_channels:int , n_time=1280) :
        super().__init__()
        self.groupnorm = nn.GroupNorm(32 , in_channels)
        self.conv = nn.Conv2d(in_channels , out_channels , kernel_size=3 , padding= 1 )
        self.linear_time = nn.Linear(n_time , out_channels)

        self.groupnorm_merged = nn.GroupNorm(32 , out_channels)
        self.conv_merged = nn.Conv2d(out_channels ,out_channels , kernel_size=3 , padding=1)

        if in_channels == out_channels :
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_channels , out_channels , kernel_size=1, padding=0)
    
    def forward(self , features , time ):
        # features : (batch_size , inchannels , height , width)
        # time : (1 ,1280)

        residue = features

        features = self.groupnorm(features)

        features = F.silu(features)

        features = self.conv(features)

        time = F.silu(time)

        time = self.linear_time(time)

        merged = features + time.unsqueeze(-1).unsqueeze(-1)
    
        merged = self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class Upsample(nn.Module):
    def __init__(self , channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels , channels , kernel_size=3 , padding=1)

    def forward(self ,x):
        x = F.interpolate(x , scale_factor=2 , mode="nearest")
        return self.conv(x)



class SwitchSequence(nn.Sequential):

    def forward(self, x: torch.Tensor , context : torch.Tensor , time: torch.Tensor)-> torch.Tensor:
        for layer in self:
            if isinstance(layer , Unet_AttentionBlock) :
                x = layer(x ,context)
            elif isinstance(layer , Unet_ResidualBlock):
                x = layer(x , time)
            else: 
                x = layer(x)

        return x 

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            #(Batch , 4 , Height/ 8 , Width/8)
            SwitchSequence(nn.Conv2d(4, 320 , kernel_size=3 , padding=1)),

            SwitchSequence(Unet_ResidualBlock(320 ,320) , Unet_AttentionBlock(8 ,40)),

            SwitchSequence(Unet_ResidualBlock(320 ,320) , Unet_AttentionBlock(8 ,40)),

            #(Batch , 320 , Height/ 8 , Width/8)->(Batch ,320 , Height/ 16 , Width/16)
            SwitchSequence(nn.Conv2d(320, 320 , kernel_size=3 , stride=2 , padding=1)),

            SwitchSequence(Unet_ResidualBlock(320 , 640) , Unet_AttentionBlock(8 ,80)),

            SwitchSequence(Unet_ResidualBlock(640 , 640) , Unet_AttentionBlock(8 ,80)),

            #(Batch , 640 , Height/ 16 , Width/16)-> (Batch , 640 , Height/ 32 , Width/32)
            SwitchSequence(nn.Conv2d(640, 640 , kernel_size=3 , stride=2 , padding=1)),

            SwitchSequence(Unet_ResidualBlock(640 , 1280) , Unet_AttentionBlock(8 ,160)),

            SwitchSequence(Unet_ResidualBlock(1280 , 1280) , Unet_AttentionBlock(8 ,160)),

            #(Batch ,1280, Height/ 32 , Width/32)-> (Batch , 1280 , Height/64  , Width/64)

            SwitchSequence(nn.Conv2d(1280, 1280 , kernel_size=3 , stride=2 , padding=1)),

            SwitchSequence(Unet_ResidualBlock(1280 , 1280)),

            SwitchSequence(Unet_ResidualBlock(1280 , 1280)),

        ])
        self.bottleneck = SwitchSequence(

            Unet_ResidualBlock(1280 , 1280),

            Unet_AttentionBlock(8 ,160),

            Unet_ResidualBlock(1280 , 1280)
        )


        self.decoders = nn.ModuleList([
            # (batch , 2560 ,Heigth / 64 , Width / 64 )-> (batch , 1280 ,Heigth / 64 , Width / 64 ) 
            SwitchSequence(Unet_ResidualBlock(2560 , 1280)),

            SwitchSequence(Unet_ResidualBlock(2560 , 1280)),

            SwitchSequence(Unet_ResidualBlock(2560 , 1280) ,Upsample(1280)),

        
            SwitchSequence(Unet_ResidualBlock(2560 , 1280) , Unet_AttentionBlock(8 ,160)),

            SwitchSequence(Unet_ResidualBlock(2560 , 1280) , Unet_AttentionBlock(8 ,160)),

            SwitchSequence(Unet_ResidualBlock(1920 , 1280) , Unet_AttentionBlock(8 ,160), Upsample(1280)),


            SwitchSequence(Unet_ResidualBlock(1920 , 640) , Unet_AttentionBlock(8 ,80)),

            SwitchSequence(Unet_ResidualBlock(1280 , 640) , Unet_AttentionBlock(8 ,80)),

            SwitchSequence(Unet_ResidualBlock(960 , 640) , Unet_AttentionBlock(8 ,80), Upsample(640)),


            SwitchSequence(Unet_ResidualBlock(960 , 320) , Unet_AttentionBlock(8 ,40)),

            SwitchSequence(Unet_ResidualBlock(640 , 320) , Unet_AttentionBlock(8 ,40)),

            SwitchSequence(Unet_ResidualBlock(640 , 320) , Unet_AttentionBlock(8 ,40)),

        ])



class Unet_outputlayer(nn.Module):
    def __inint__(self ,in_channels:int , out_channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels ,kernel_size= 3 , padding= 1)
    
    def forward(self , x ):

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        return x 

class Diffusion(nn.Module): 

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = Unet()
        self.final = Unet_outputlayer(320 , 4 )

    def forward(self , latent , context , time): 
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

         # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        #(Batch , 4 , Height/ 8 , Width/8)-> (Batch , 320 , Height / 8 , Width/8)
        output = self.unet(latent , context , time)

        #(Batch , 320, Height/ 8 , Width/8)-> (Batch , 4 , Height / 8 , Width/8)
        output = self.final(output)

        return output