import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.d_k=getattr(config,"d_k",128)
        self.group_size=int(math.sqrt(config.block_size))
        assert self.group_size * self.group_size == config.block_size, "block_size must be a perfect square"
        self.num_groups=config.block_size // self.group_size

        self.Wq=nn.Linear(config.n_embd,self.d_k, bias=False)
        self.Wk=nn.Linear(config.n_embd,self.d_k, bias=False)
        self.Wv=nn.Linear(config.n_embd,self.d_k, bias=False)

        self.c_proj=nn.Linear(self.d_k, config.n_embd, bias=config.bias)
        self.attn_dropout=nn.Dropout(config.dropout)
        self.resid_dropout=nn.Dropout(config.dropout)
        self.n_head=config.n_head

    def forward(self,X):
        Batch,Token,Embedding=X.shape                                        # X: (B, T, D)                 e.g. (B, 1024, 768)
        assert Token % self.group_size == 0
        num_groups=Token // self.group_size

        # Level 1 - local attention per group
        G=X.view(Batch,num_groups,self.group_size,Embedding)                 # G: (B, g, s, D)               e.g. (B, 32, 32, 768)
        Q_local=self.Wq(G)                                                   # Q_local: (B, g, s, d_k)       e.g. (B, 32, 32, 128)
        K_local=self.Wk(G)                                                   # K_local: (B, g, s, d_k)       e.g. (B, 32, 32, 128)
        a=(Q_local @ K_local.transpose(-2,-1)) * (1.0/math.sqrt(self.d_k))   # a: (B, g, s, s)               e.g. (B, 32, 32, 32)   [Q_local @ K_local^T]
        a=F.softmax(a,dim=-1)                                                # a: (B, g, s, s)   (unchanged shape, softmax over last dim)
        a=self.attn_dropout(a)                                               # a: (B, g, s, s)   (unchanged shape)

        V=self.Wv(X)                                                         # V: (B, T, d_k)                e.g. (B, 1024, 128)
        V_groups=V.view(Batch,num_groups,self.group_size,self.d_k)           # V_groups: (B, g, s, d_k)      e.g. (B, 32, 32, 128)
        raw_local=a @ V_groups                                               # raw_local: (B, g, s, d_k)     e.g. (B, 32, 32, 128)

        g=raw_local.sum(dim=2)                                               # g: (B, g, d_k)                e.g. (B, 32, 128)      [additive squish over s]
        G_prime=self.c_proj(g)                                               # G_prime: (B, g, D)            e.g. (B, 32, 768)

        # Level 2 - global inter-group attention
        Q_global=self.Wq(G_prime)                                            # Q_global: (B, g, d_k)         e.g. (B, 32, 128)
        K_global=self.Wk(G_prime)                                            # K_global: (B, g, d_k)         e.g. (B, 32, 128)
        a_prime=(Q_global @ K_global.transpose(-2,-1)) * (1.0/math.sqrt(self.d_k))  # a_prime: (B, g, g)     e.g. (B, 32, 32)
        a_prime=F.softmax(a_prime,dim=-1)                                    # a_prime: (B, g, g)   (unchanged shape)
        a_prime=self.attn_dropout(a_prime)                                   # a_prime: (B, g, g)   (unchanged shape)

        # Combine local and global attention
        weighted=a_prime.unsqueeze(-1).unsqueeze(-1) * a.unsqueeze(1)        # a_prime.unsqueeze(-1).unsqueeze(-1): (B, g, g, 1, 1) e.g. (B,32,32,1,1)
                                                                              # a.unsqueeze(1):                       (B, 1, g, s, s) e.g. (B,1,32,32,32)
                                                                              # weighted (broadcast product):         (B, g_out=i, g_in=j, s_q=r, s_k=c) e.g. (B, 32, 32, 32, 32)
        t=weighted.permute(0,1,3,2,4).reshape(Batch,num_groups,self.group_size,Token)
                                                                              # permute(0,1,3,2,4): (B, i, r, j, c)   e.g. (B, 32, 32, 32, 32)
                                                                              # reshape (flatten j,c -> T):  t: (B, g, s, T)  e.g. (B, 32, 32, 1024)

        y=t @ V.unsqueeze(1)                                                 # V.unsqueeze(1): (B, 1, T, d_k) e.g. (B,1,1024,128)
                                                                              # y = t @ V:      (B, g, s, d_k) e.g. (B, 32, 32, 128)
        y=self.c_proj(y)                                                     # y: (B, g, s, D)                e.g. (B, 32, 32, 768)
        y=y.reshape(Batch,Token,Embedding)                                   # y: (B, T, D)                   e.g. (B, 1024, 768)
        y=self.resid_dropout(y)                                              # y: (B, T, D)   (unchanged shape)
        return y
