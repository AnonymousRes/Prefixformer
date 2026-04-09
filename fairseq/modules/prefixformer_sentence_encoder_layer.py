# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
from fairseq import utils
from fairseq.modules.sequence_norm import SequenceNorm
from fairseq.modules import (
    FairseqDropout
)


def apply_attention(query, key, value, attn_padding_mask=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(query.size(-1))
    if attn_padding_mask is not None:
        scores = scores.masked_fill(attn_padding_mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    return torch.matmul(p_attn, value), p_attn


def prefix_softmax_weighted_sum(x, prefix_logits):
    """
    x:             (B, L, D)
    prefix_logits: (B, L, 1)
    return:        prefix_x (B, L, D)
    """
    # (B, L)
    logits = prefix_logits.squeeze(-1)

    prefix_max = logits.cummax(dim=1).values  # (B, L)

    exp_x = torch.exp(logits - prefix_max)

    prefix_sum = exp_x.cumsum(dim=1)  # (B, L)

    weighted_prefix = (exp_x.unsqueeze(-1) * x).cumsum(dim=1)  # (B, L, D)

    prefix_x = weighted_prefix / prefix_sum.unsqueeze(-1)      # (B, L, D)

    return prefix_x


class DynamicAdapterMultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, dropout=0.1, shared_generator=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.d_k = embedding_dim // num_attention_heads
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)
        self.att_dropout = nn.Dropout(p=dropout)
        self.shared_generator = shared_generator
        self.k_layer_bias = nn.Parameter(torch.zeros(self.embedding_dim))
        self.v_layer_bias = nn.Parameter(torch.zeros(self.embedding_dim))
        self.k_scale = nn.Parameter(torch.ones(1))
        self.v_scale = nn.Parameter(torch.ones(1))


    def forward(self, query, key, value, self_attn_padding_mask=None):
        B, L, D = query.size()

        key_ = self.k_scale * key + self.k_layer_bias
        value_ = self.v_scale * value + self.v_layer_bias

        query = query.view(B, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        key_ = key_.view(B, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
        value_ = value_.view(B, -1, self.num_attention_heads, self.d_k).transpose(1, 2)

        x, attn = apply_attention(query, key_, value_, self_attn_padding_mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.embedding_dim)
        return self.att_dropout(self.output_linear(x)), key, value, attn


class Prefix_Layer(nn.Module): #1027
    def __init__(self, embedding_dim: int = 128, b_count: int = 0, b_size: int = 0):
        super().__init__()
        self.block_count = b_count
        self.block_size = b_size
        self.embedding_dim = embedding_dim

        self.linear_w = nn.Linear(embedding_dim, 1)

        self.k_size_main = 6
        self.main_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=embedding_dim,
                kernel_size=self.k_size_main,
                padding='same'
            ),
            nn.Dropout(0.1),
        )

        self.overlap_size = self.block_count * 2
        self.linear_fusion_attout = nn.Linear(3*embedding_dim, embedding_dim)
        self.fusion_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor):
        """
        x: [B, L, D]
        Returns: out, out
        """
        B, L, D = x.shape
        L_pad = self.block_count * self.block_size
        x = F.pad(x, (0,0,0, L_pad-L), value=0.0)

        prefix_logits = F.silu(self.linear_w(x))  # (B, L, 1)
        prefix_x = prefix_softmax_weighted_sum(x, prefix_logits) + x

        prefix_x_blocks = prefix_x.view(B*self.block_count, self.block_size, D)
        prefix_x_blocks = self.main_conv(prefix_x_blocks.transpose(1,2)).transpose(1,2)
        prefix_block_repr = torch.max(prefix_x_blocks, dim=1)[0]
        prefix_block_repr = prefix_block_repr.view(B*self.block_count, 1, D)

        x_overlap = F.pad(prefix_x, (0,0,0, self.overlap_size//2), mode='replicate')[:, self.block_size-self.overlap_size//2:, :]
        x_blocks_overlap = x_overlap.unfold(dimension=1,size=self.overlap_size, step=self.block_size) #[B, block_count, D, overlap_size]
        x_blocks_overlap = x_blocks_overlap.permute(0, 1, 3, 2).contiguous().view(B*self.block_count, self.overlap_size, D)

        b2s_attn_out, _ = apply_attention(prefix_block_repr, prefix_x_blocks, prefix_x_blocks)
        b2o_attn_out, _ = apply_attention(prefix_block_repr, x_blocks_overlap, x_blocks_overlap)
        outfusion = torch.cat([b2s_attn_out, b2o_attn_out, prefix_block_repr], dim=-1)
        out = self.linear_fusion_attout(outfusion).view(B, self.block_count, D)

        return out, out


class PrefixformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """
    def __init__(
        self,
        embedding_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        norm_type='layernorm',
        export: bool = False,
        init_fn: Callable = None,
        shared_generator = None
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = activation_fn # silu
        self.activation_fn = utils.get_activation_fn(self.activation_fn)
        self.shared_generator = shared_generator
        self.self_attn = DynamicAdapterMultiheadAttention(self.embedding_dim, num_attention_heads, dropout,
                                                          self.shared_generator)
        self.norm_type = norm_type
        self.self_attn_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, affine=True, export=export)
        self.fc1 = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
        self.final_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, affine=True, export=export)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, key, value, attn = self.self_attn(
            query=x,
            key=k,
            value=v,
            self_attn_padding_mask=self_attn_padding_mask
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, key, value, attn