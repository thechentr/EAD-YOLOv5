import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        num_blocks,
        embedding_dim,
        residual_pdrop,
        attention_pdrop
    ):
        super(CausalSelfAttention, self).__init__()
        assert embedding_dim % num_heads == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        # regularization
        self.attn_drop = nn.Dropout(attention_pdrop)
        self.resid_drop = nn.Dropout(residual_pdrop)
        # output projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.ones(num_blocks, num_blocks).view(1, 1, num_blocks, num_blocks))
        self.num_heads = num_heads

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, num_heads, num_blocks, embedding_dim, residual_pdrop, attention_pdrop):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(num_heads, num_blocks, embedding_dim, residual_pdrop, attention_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        num_blocks,
        residual_pdrop,
        attention_pdrop,
        embedding_pdrop,
        embedding_dim,
        output_dim,
    ):
        super(GPT, self).__init__()
        # input embedding stem
        self.seq_pos_embed = nn.Parameter(torch.zeros(1, num_blocks, embedding_dim), requires_grad=True)
        self.embed_dim = embedding_dim

        # assert num_blocks == (num_img_blocks + num_act_blocks), \
        #        "number of blocks inconsistent"
        self.num_blocks = num_blocks
        # self.num_img_blocks = num_img_blocks
        # self.num_act_blocks = num_act_blocks

        self.drop = nn.Dropout(embedding_pdrop)
        # transformer
        args = (num_heads, num_blocks, embedding_dim, residual_pdrop, attention_pdrop)
        self.blocks = nn.Sequential(*[Block(*args) for _ in range(num_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, output_dim, bias=False)

        self.apply(self._init_weights)
        self.init_pos_emb()

    def init_pos_emb(self):
        # Embedding type (x2): image feature embedding, action embedding 
        # task_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, 2)
        # task_pos_embed = torch.zeros((1, 2, self.embed_dim)) # [1,2,768/2]
        # task_pos_embed[:, :, self.embed_dim // 2:] = torch.from_numpy(task_sin_embed).float()
        # # [1,2,768/2~]

        # img_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, self.num_img_blocks)
        # img_pos_embed = torch.zeros((1, self.num_img_blocks, self.embed_dim))# [1,4,768/2]
        # img_pos_embed[:, :, :self.embed_dim // 2] = torch.from_numpy(img_sin_embed).float()
        # # [1,4,~768/2]
        # action_sin_embed = get_1d_sincos_pos_embed(self.embed_dim // 2, self.num_act_blocks)
        # action_pos_embed = torch.zeros((1, self.num_act_blocks, self.embed_dim))# [1,1,768/2]
        # action_pos_embed[:, :, :self.embed_dim // 2] = torch.from_numpy(action_sin_embed).float()
        # # [1,1,~768/2]

        # pos_emb = torch.zeros((1, self.num_blocks, self.embed_dim))# [1,5,768/2]
        # pos_emb[:, :self.num_img_blocks] = img_pos_embed + task_pos_embed[:, 0, :]
        # # [1,~4,768/2] = # [1,4,~768/2] + # [1,1,768/2~]
        # pos_emb[:, self.num_img_blocks:] = action_pos_embed + task_pos_embed[:, 1, :]
        sin_embed = get_1d_sincos_pos_embed(self.embed_dim, self.num_blocks)
        pos_embed = torch.zeros((1, self.num_blocks, self.embed_dim)) # [1,2,768/2]
        pos_embed[:, :, :] = torch.from_numpy(sin_embed).float()
        self.seq_pos_embed.data.copy_(pos_embed)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, seq):
        b, t = seq.shape[:2] # [B, S+1, F]
        assert t <= self.num_blocks, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        position_embeddings = self.seq_pos_embed[:, -t:, :] # each position maps to a (learnable) vector
        x = self.drop(seq + position_embeddings) # always add the action token and the last frame token with the same position_embeddings
        x = self.blocks(x) # several self attention block layers
        x = self.ln_f(x) # a linear layer
        logits = self.head(x)

        return logits # [B, S+1, F]


# Positional embeddings
def get_1d_sincos_pos_embed(embed_dim, n):
    grid = np.arange(n)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
