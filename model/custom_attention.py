import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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
        # Memory note: the combine step materializes 'weighted' (B,g,g,s,s) and 't' (B,g,s,T) — together ~2M
        # elements per sample (e.g. 2 x (B,32,32,32,32)-ish), which is the same order as a full T x T = 1024x1024
        # attention matrix. Keeping them alive for backprop would defeat the point of hierarchical attention.
        # So during training the combine is wrapped in torch.utils.checkpoint: after the forward pass only the
        # inputs (a = locals, a_prime = globals, V) are saved, 'weighted' and 't' are freed immediately, and they
        # are regenerated on the fly during backprop. Costs one extra forward of this (cheap) segment; saves the
        # two big tensors per layer.
        if self.training and torch.is_grad_enabled():
            y=checkpoint(self._combine, a, a_prime, V, use_reentrant=False)  # y: (B, g, s, d_k)             e.g. (B, 32, 32, 128)
        else:
            y=self._combine(a, a_prime, V)                                   # y: (B, g, s, d_k)             e.g. (B, 32, 32, 128)
        y=self.c_proj(y)                                                     # y: (B, g, s, D)                e.g. (B, 32, 32, 768)
        y=y.reshape(Batch,Token,Embedding)                                   # y: (B, T, D)                   e.g. (B, 1024, 768)
        y=self.resid_dropout(y)                                              # y: (B, T, D)   (unchanged shape)
        return y

    def _combine(self, a, a_prime, V):
        # Recomputable segment (checkpointed during training): builds the hierarchical attention rows t and
        # applies them to V. Inputs are only the locals (a), globals (a_prime) and values (V); everything
        # created inside is freed after forward and regenerated on the fly during backprop.
        Batch,num_groups,group_size,_=a.shape
        Token=V.shape[1]
        # Concrete toy example: T=8 tokens, num_groups=g=2, group_size=s=4 (so g*s=T=8), batch dropped (b fixed).
        #   Group 0 = tokens[0:4], Group 1 = tokens[4:8].
        #   a has shape (g=2, s=4, s=4):  a[0] = a_0 (group 0's own 4x4 local attention matrix), a[1] = a_1 (group 1's own 4x4 local attention matrix)
        #     a_0 = [[1,0,0,0],
        #            [0,1,0,0],
        #            [0,0,1,0],
        #            [0,0,0,1]]
        #     a_1 = [[.25,.25,.25,.25],
        #            [.25,.25,.25,.25],
        #            [.25,.25,.25,.25],
        #            [.25,.25,.25,.25]]
        #   a_prime has shape (g=2, g=2):  a_prime = [[0.7, 0.3], [0.4, 0.6]]   (a_prime[i,j] = how much output group i attends to group j)
        #   Goal for output group i=0: scale each FULL local matrix by the scalar a_prime[0,j], then concat side by side (horizontally):
        #     t_0 = [ a_prime[0,0]*a_0 , a_prime[0,1]*a_1 ]
        #         = [ 0.7*[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] , 0.3*[[.25,.25,.25,.25],[.25,.25,.25,.25],[.25,.25,.25,.25],[.25,.25,.25,.25]] ]
        #     which multiplies out to the (s=4, T=8) matrix:
        #         [[0.7,0,0,0,   .075,.075,.075,.075],
        #          [0,0.7,0,0,   .075,.075,.075,.075],
        #          [0,0,0.7,0,   .075,.075,.075,.075],
        #          [0,0,0,0.7,   .075,.075,.075,.075]]
        #     (each row sums to 1, e.g. row0: 0.7 + 4*0.075 = 1.0)
        weighted=a_prime.unsqueeze(-1).unsqueeze(-1) * a.unsqueeze(1)        # a_prime.unsqueeze(-1).unsqueeze(-1): (g,g,1,1) = (2,2,1,1)  -> a_prime[i,j] broadcast-ready to scale a WHOLE (r,c) matrix
                                                                              # a.unsqueeze(1):                       (1,g,s,s) = (1,2,4,4)  -> a[j] (each full 4x4 matrix) broadcast-ready over i
                                                                              # weighted (broadcast product):         (g_out=i, g_in=j, s_q=r, s_k=c) = (2,2,4,4)
                                                                              # weighted[i,j,:,:] = a_prime[i,j] * a[j,:,:]   <- for fixed i,j this is the WHOLE scaled matrix a_prime[i,j]*a_j, not just one row
                                                                              # for i=0: weighted[0,0,:,:] = 0.7*a_0 = [[0.7,0,0,0],[0,0.7,0,0],[0,0,0.7,0],[0,0,0,0.7]]
                                                                              #          weighted[0,1,:,:] = 0.3*a_1 = [[.075,.075,.075,.075], ... (4 identical rows)]
                                                                              # i.e. fixing i=0, varying j=0,1 gives exactly the 2 bracketed FULL matrices of t_0 above, just stacked on a separate j axis (not concatenated yet)
        t=weighted.permute(0,1,3,2,4).reshape(Batch,num_groups,group_size,Token)
                                                                              # permute(0,1,3,2,4): (B,i,r,j,c) = (B,2,4,2,4)  -> moves j,c next to each other, in (j,c) order
                                                                              # reshape (flatten j,c -> T=8):  t: (B,g,s,T) = (B,2,4,8)
                                                                              # flattening (j,c) with j-major, c-minor horizontally concatenates the FULL matrices column-wise:
                                                                              #   t[b,0,:,:] = [ weighted[b,0,0,:,:] | weighted[b,0,1,:,:] ]  (side by side, 4x4 next to 4x4 -> 4x8)
                                                                              #              = [[0.7,0,0,0,  .075,.075,.075,.075],
                                                                              #                 [0,0.7,0,0,  .075,.075,.075,.075],
                                                                              #                 [0,0,0.7,0,  .075,.075,.075,.075],
                                                                              #                 [0,0,0,0.7,  .075,.075,.075,.075]]     <- matches t_0 computed above exactly
                                                                              # this IS t_0, the full (s=4, T=8) matrix for output group i=0 (reading out row r gives the earlier per-row formula, they're the same tensor)

        y=t @ V.unsqueeze(1)                                                 # V.unsqueeze(1): (B, 1, T, d_k) e.g. (B,1,1024,128)
                                                                              # y = t @ V:      (B, g, s, d_k) e.g. (B, 32, 32, 128)
        return y
