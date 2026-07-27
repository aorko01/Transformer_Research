import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.d_k=getattr(config,"d_k",128)
        self.num_groups=getattr(config,"num_groups",32)
        assert config.block_size % self.num_groups == 0
        self.group_size=config.block_size // self.num_groups

        self.Wq=nn.Linear(config.n_embd,self.d_k, bias=False)
        self.Wk=nn.Linear(config.n_embd,self.d_k, bias=False)
        self.Wv=nn.Linear(config.n_embd,self.d_k, bias=False)

        self.c_proj=nn.Linear(self.d_k, config.n_embd, bias=config.bias)
        self.attn_dropout=nn.Dropout(config.dropout)
        self.resid_dropout=nn.Dropout(config.dropout)
        self.n_head=config.n_head

    def forward(self,X):
        Batch,Token,Embedding=X.shape
        assert Token % self.group_size == 0
        num_groups=Token // self.group_size

        # Level 1 - local attention per group
        G=X.view(Batch,num_groups,self.group_size,Embedding)
        Q_local=self.Wq(G)
        K_local=self.Wk(G)
        a=(Q_local @ K_local.transpose(-2,-1)) * (1.0/math.sqrt(self.d_k))
        a=F.softmax(a,dim=-1)
        a=self.attn_dropout(a)

        V=self.Wv(X)
        V_groups=V.view(Batch,num_groups,self.group_size,self.d_k)
        raw_local=a @ V_groups

        g=raw_local.sum(dim=2)
        G_prime=self.c_proj(g)

        # Level 2 - global inter-group attention
        Q_global=self.Wq(G_prime)
        K_global=self.Wk(G_prime)
        a_prime=(Q_global @ K_global.transpose(-2,-1)) * (1.0/math.sqrt(self.d_k))
        a_prime=F.softmax(a_prime,dim=-1)
        a_prime=self.attn_dropout(a_prime)

        # Combine local and global attention
        weighted=a_prime.unsqueeze(-1).unsqueeze(-1) * a.unsqueeze(1)
        t=weighted.permute(0,1,3,2,4).reshape(Batch,num_groups,self.group_size,Token)

        y=t @ V.unsqueeze(1)
        y=self.c_proj(y)
        y=y.reshape(Batch,Token,Embedding)
        y=self.resid_dropout(y)
        return y
