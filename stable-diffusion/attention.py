import torch 
from torch import nn 
import torch.nn.functional as F 
import math 

class SelfAttention(nn.Module):
    def __init__(self , n_heads:int , d_embed:int , input_proj_bias=True , output_proj_bias=True):
        super().__init__()

        self.input_proj = nn.Linear(d_embed , d_embed * 3 , bias=input_proj_bias)
        self.output_proj = nn.Linear(d_embed , d_embed , bias=output_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self , x:torch.Tensor , causal_mask=False) -> torch.Tensor:
        # x : (batch_size , seq_len , dim)
        input_shape = x.shape 
        
        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape 

        intermidiate_shape = (batch_size , sequence_length , self.n_heads ,self.d_head)

        # (batch_size , seq_len , dim )--> (batch_size ,seq_len , dim *3)--> 3 tensors of shape(batch_size , seq_len , dim)
        query ,key ,value = self.input_proj(x).chunk(3 , dim = -1)

        # ( batch_size . seq_len , dim)-->(batch_size, seq_len , H , dim/H)-->(batch_size, seq_len ,dim/ H , H)
        query = query.view(intermidiate_shape).transpose( 1, 2)
        key = key.view(intermidiate_shape).transpose( 1, 2)
        value = value.view(intermidiate_shape).transpose( 1, 2)

        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        attention= query @ key.transpose(-1, -2)
        
        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(attention, dtype=torch.bool).triu(1) 
            # Fill the upper triangle with -inf
            attention.masked_fill_(mask, -torch.inf) 
        
        # Divide by d_k (Dim / H). 
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        atttention /= math.sqrt(self.d_head) 

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        attention = F.softmax(attention, dim=-1) 

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = attention @ value

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.view(input_shape) 

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.output_proj(output) 
        
        # (Batch_Size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self , n_heads: int , d_embed : int , d_cross: int , input_proj_bias= True , output_proj_bias=True):
        super().__init__()
        self.query_proj = nn.Linear(d_embed , d_embed, bias=input_proj_bias)
        self.key_proj = nn.Linear(d_cross , d_embed, bias=input_proj_bias)
        self.value_proj = nn.Linear(d_cross , d_embed, bias=input_proj_bias)
        self.output_proj = nn.Linear(d_embed , d_embed,bias=output_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self , x , y):
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)
        input_shape = x.shape 
        batch_size, sequence_length , d_embed = input_shape
        intermidiate_shape = (batch_size , -1 , self.n_heads ,self.d_head )

        query = self.query_proj(x)
        key = self.key_proj(y)
        value= self.value_proj(y)

        query = query.view(intermidiate_shape).transpose(1,2)
        key = key.view(intermidiate_shape).transpose(1,2)
        value = value.view(intermidiate_shape).transpose(1,2)

        attention = query @ key.transpose(-1, -2)
        attention /= math.sqrt(self.d_head)
        attention = F.softmax(attention, dim=-1)

        output = attention @ value
        output = output.transpose(1,2).contiguous()
        output = output.view(input_shape)

        output = self.output_proj(output)   

        return output