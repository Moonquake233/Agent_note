import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,dropout:float = 0.1,max_len:int = 5000): #max_len相当于一次消息最长的token长度，d_model是模每个 token 用多少维向量表示
        super().__init__()
        self.dropout = nn.Dropout(p = dropout)
        #创建足够长的位置编码           unsqueeze(i) 在第i位置插入一个维度
        position = torch.arange(max_len).unsqueeze(1) # torch.arange(5000) 得到 [0,1,2,...,4999]  unsqueeze(1) 把它变成 (5000, 1)
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe = torch.zeros(max_len,d_model)   
        pe[:,0::2] = torch.sin(position*div_term)           # a[start:stop:step]  [,] 逗号用来区分维度， pe[:, 0::2] 它是两份方案：第 0 维的方案是 :（全要），第 1 维的方案是 0::2（取偶数位）。
        pe[:,1::2] = torch.cos(position*div_term)

        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x:torch.Tensor) -> torch.Tensor:   #x是embedding
        x = x+self.pe[:,:x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_heads==0 #d_model 必须要被num_heads整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads #注意力头的数量

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self,Q,K,V,mask = None):
        attn_scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)

        #掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        atten_probs = torch.softmax(attn_scores,dim=-1)
        output = torch.matmul(atten_probs,V)
        return output

    def split_heads(self,x):
        #将输入的形状从 batch_size，seq_length，d_model 变为 batch_size，num_heads,seq_length，d_k
        batch_size,seq_length,d_model = x.size()
        return x.view(batch_size,seq_length,self.num_heads,self.d_k).transpose(1,2)         
    
    def combine_heads(self,x):
        batch_size,num_heads,seq_length,d_K = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_length,self.d_model)
    
    def forward(self,Q,K,V,mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q,K,V,mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Encoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(Encoder,self).__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward =FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output = self.self_attn(x,x,x,mask)
        x= self.norm1(x+self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x+self.dropout(ff_output))

        return x


class Decoder(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(Decoder,self).__init__()
        self.self_attn = MultiHeadAttention(d_model,num_heads)
        self.cross_attn = MultiHeadAttention(d_model,num_heads)
        self.feed_forward = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.droput = nn.Dropout(dropout)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        attn_output = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x+self.droput(attn_output))

        cross_attn_output = self.cross_attn(x,encoder_output,encoder_output,src_mask)
        x =self.norm2(x+self.droput(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x+self.droput(ff_output))

        return x

def make_pad_mask(seq, pad_idx=0):  
    """(B, T) -> (B, 1, 1, T)。 pad 是填充字符"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size, device=None):
    """(T, T) 下三角。True = 允许看，False = 未来位置（不许看）。"""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))  #torch.ones(size, size) 是全 1 方阵，torch.tril 取左下三角（含对角线）​，其余置 0。


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, dropout=0.1, max_len=5000, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionEncoder(d_model, dropout, max_len)

        self.encoder_layers = nn.ModuleList(
            [Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  #[表达式 for _ in range(n)] 的规则是"把表达式执行 n 次，结果装进一个 list"。
        self.decoder_layers = nn.ModuleList(
            [Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc_out = nn.Linear(d_model, tgt_vocab)   # d_model -> 词表大小

    def encode(self, src, src_mask):
        x = self.pos_enc(self.src_embed(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, src_mask, tgt_mask):
        x = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = make_pad_mask(src, self.pad_idx)                              # (B,1,1,T_src)
        tgt_pad = make_pad_mask(tgt, self.pad_idx)                               # (B,1,1,T_tgt)
        tgt_mask = tgt_pad & make_causal_mask(tgt.size(1), tgt.device)           # 解码器的自注意力要同时满足两个条件：不是 pad，并且不是未来位置。

        encode_output = self.encode(src, src_mask)
        out = self.decode(tgt, encode_output, src_mask, tgt_mask)
        return self.fc_out(out)      # (B, T_tgt, tgt_vocab)，还没 softmax


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_src, T_tgt = 2, 7, 5   # T_Src和T_tgt 对应的是seq_length
    src_vocab, tgt_vocab = 100, 120

    model = Transformer(src_vocab, tgt_vocab, d_model=32, num_heads=4,
                        d_ff=64, num_layers=2, dropout=0.1)

    src = torch.randint(1, src_vocab, (B, T_src))  # randint(low, high, size) 左闭右开，取值 [1, 99]
    tgt = torch.randint(1, tgt_vocab, (B, T_tgt))
    src[0, -2:] = 0          # 手动造两个 padding
    tgt[1, -1:] = 0

    logits = model(src, tgt)
    print("logits.shape =", tuple(logits.shape))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    target = torch.randint(1, tgt_vocab, (B, T_tgt))  #target 是假的"正确答案"。真实训练时它是答案 token 的 id。
    loss = criterion(logits.reshape(-1, tgt_vocab), target.reshape(-1))
    print("loss =", loss.item())

    n_params = sum(p.numel() for p in model.parameters())
    print("参数量 =", n_params)

    print("梯度反传 ...")
    loss.backward()
    print("第一个 W_q.weight 梯度范数 =",
          model.encoder_layers[0].self_attn.W_q.weight.grad.norm().item())