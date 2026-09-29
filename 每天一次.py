import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int = 5000,dropout:float = 0.1):  
        super().__init__()
        dropout = nn.Dropout(p = dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000)/d_model))
        pe = torch.zeros(max_len,d_model)
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        self.register_buffer = ("pe",pe.unsqueeze(0))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = x +self.pe[:,x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff,dropout = 0.1):
        super().__init__()
        self.Linear1 = nn.Linear(d_model,d_ff)
        self.Linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        def forward(self,x):
            x = self.Linear1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.Linear2(x)
            return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        assert d_model % num_heads ==0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self,K,Q,V,mask = None):
        attn_scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask ==0,-1e9)
        attn_probs = torch.softmax(attn_scores,dim=-1)
        output = torch.matmul(attn_probs,V)
        return output

    def split_heads(self,x):
        batch_size,seq_length,d_model = x.size()
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)
    def combine_heads(self,x):
        batch_size,num_heads,seq_length,d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)

    def forward(self,Q,K,V,mask= None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_q(K))
        V = self.split_heads(self.W_q(V))

        attn_output = self.scaled_dot_product_attention(Q,K,V,mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model=d_model,d_ff=d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output = self.attn(x,x,x,mask)
        x = self.LayerNorm1(x+self.dropout(attn_output))

        feed_output = self.feedforward(x)

        x = self.LayerNorm2(x+self.dropout(feed_output))

        return x

class Decoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.cross_attn = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)
        self.LayerNorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = FeedForward(d_model,d_ff)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        self_attn = self.self_attn(x,x,x,tgt_mask)
        x = self.LayerNorm1(x+self.dropout(self_attn))

        corss_attn = self.cross_attn(encoder_output,encoder_output,x,src_mask)
        x = self.LayerNorm2(x+self.dropout(self.cross_attn))

        ff_output = self.feedforward(x)
        x = self.LayerNorm3(x+self.dropout(ff_output))
        return x

