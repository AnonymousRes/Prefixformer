# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
from fairseq.modules.sequence_norm import SequenceNorm
from fairseq.modules.prefixformer_sentence_encoder_layer import PrefixformerSentenceEncoderLayer, SharedQKVGenerator
from fairseq.modules.prefixformer_sentence_encoder_layer import Prefix_Layer
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    RealNumberEmbedding
)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class PrefixformerSCEncoder(nn.Module):
    """
    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        num_encoder_layers: int = 6,
        embedding_dim: int = 512,
        ffn_hidden_dim: int = 1024,
        num_attention_heads: int = 8,
        ktimes: int = 1,
        activation: str = 'silu',
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_type: str = 'layernorm',
        normalize_before: bool = False,
        layerdrop: float = 0.0,
        max_seq_len: int = 16000,
        export: bool = False,
        traceable: bool = False,
        sen_rep_type: str = 'mp',
    ) -> None:

        super().__init__()
        self.embedding_dropout = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = dropout
        self.max_seq_len = max_seq_len
        self.ktimes = ktimes
        self.block_count = math.ceil(math.log2(self.max_seq_len)) * self.ktimes
        self.block_size = math.ceil(self.max_seq_len / self.block_count)
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.sen_rep_type = sen_rep_type

        self.embed_tokens = RealNumberEmbedding(self.embedding_dim)
        self.pfcl = PrefixFourierConv_Layer(embedding_dim=self.embedding_dim, b_count=self.block_count,
                                            b_size=self.block_size)
        self.shared_generator = SharedQKVGenerator(d_model=self.embedding_dim)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers

        self.layers.extend([
            PrefixformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=self.embedding_dim,
                num_attention_heads=self.num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=dropout,
                activation_fn=activation,
                norm_type=norm_type,
                export=export,
                shared_generator=self.shared_generator
            )
            for _ in range(self.num_layers)
        ])

        if normalize_before:
            self.final_norm = SequenceNorm(norm_type, embedding_dim, export=export)
        else:
            self.final_norm = None

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:

        bsz, seq_len = tokens.size()

        padding_mask = None
        # B x T -> B x T x D
        x = self.embed_tokens(tokens)
        x = self.embedding_dropout(x)

        # B x T x C
        key, value = self.pfcl(x)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for i in range(self.num_layers):
            x, key, value, _ = self.layers[i](x=x, k=key, v=value)
            if not last_state_only:
                inner_states.append(x)

        if self.final_norm is not None:
            x = self.final_norm(x)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=-2) / src_lengths.unsqueeze(1)
        else:
            sentence_rep = x[0, :, :]
        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep