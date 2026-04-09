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
from fairseq.modules.prefixformer_sentence_encoder_layer import Prefix_Layer, Prefix_Layer_a1, Prefix_Layer_a2
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    PositionalEmbedding,
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


class PrefixformerLRAEncoder(nn.Module):
    """
    Implementation for a Prefixformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    PrefixformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_type: str = "sparse",
            embedding_dim: int = 768,
            num_attention_heads: int = 8,
            ktimes: int = 1,
            dropout: float = 0.1,
            activation_dropout: float = 0.1,
            max_seq_len: int = 256,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            embed_scale: float = None,
            norm_type: str = 'layernorm',
            export: bool = False,
            traceable: bool = False,
            tie_layer_weights: bool = False,
            sen_rep_type: str = 'mp'
    ) -> None:

        super().__init__()
        self.sen_rep_type = sen_rep_type
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = 0.0
        self.max_seq_len = max_seq_len
        self.ktimes = ktimes
        self.block_count = math.ceil(math.log2(self.max_seq_len)) * self.ktimes
        self.block_size = math.ceil(self.max_seq_len / self.block_count)
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.tie_layer_weights = tie_layer_weights
        self.norm_type = norm_type

        assert embedding_type in ['sparse', 'linear']
        self.embed_tokens = self.build_embedding(self.embedding_type, self.vocab_size, self.embedding_dim,
                                                 self.padding_idx)
        self.embed_scale = embed_scale

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )
        if self.use_position_embeddings and not self.learned_pos_embedding:
            if self.embed_scale is None:
                self.embed_scale = math.sqrt(self.embedding_dim)

        self.pfcl = Prefix_Layer(embedding_dim=self.embedding_dim, b_count=self.block_count, b_size=self.block_size)
        # self.pfcl = Prefix_Layer_a1(embedding_dim=self.embedding_dim, b_count=self.block_count, b_size=self.block_size)
        # self.pfcl = Prefix_Layer_a2(embedding_dim=self.embedding_dim, b_count=self.block_count, b_size=self.block_size)
        # self.shared_generator = SharedQKVGenerator(d_model=self.embedding_dim)
        self.shared_generator = None

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.num_layers = num_encoder_layers
        if self.tie_layer_weights:
            real_num_layers = 1
        else:
            real_num_layers = num_encoder_layers

        # self.shared_adapter = SharedDynamicKVAdapter(embed_dim=self.embedding_dim, block_count=self.block_count, block_size=self.block_size)
        self.layers.extend([
            PrefixformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                num_attention_heads=self.num_attention_heads,
                dropout=dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                norm_type=norm_type,
                export=export,
                shared_generator=self.shared_generator
            )
            for _ in range(real_num_layers)
        ])
        if encoder_normalize_before:
            self.emb_layer_norm = SequenceNorm(self.norm_type, self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None
        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

    def build_embedding(self, embedding_type, vocab_size, embedding_dim, padding_idx):
        if embedding_type == 'sparse':
            embed_tokens = nn.Embedding(vocab_size, embedding_dim, padding_idx)
            nn.init.normal_(embed_tokens.weight, mean=0, std=embedding_dim ** -0.5)
            return embed_tokens
        else:
            embed_tokens = RealNumberEmbedding(embedding_dim)
            return embed_tokens

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
            self,
            tokens: torch.Tensor,
            src_lengths: torch.Tensor,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]:
        # print('USING Prefixformer')
        # print('Before embedding:', tokens.shape)
        # compute padding mask. This is needed for multi-head attention
        # print(torch.max(tokens),torch.min(tokens))
        if self.embedding_type == 'sparse':
            padding_mask = tokens.eq(self.padding_idx)
            if not self.traceable and not self.tpu and not padding_mask.any():
                padding_mask = None
            # B x T -> B x T x D
            x = self.embed_tokens(tokens)
            # print('sparse embedding:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        else:
            padding_mask = None
            # B x T -> B x T x 1 -> B x T x D
            x = self.embed_tokens(tokens)
            # print('no sparse embedding:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        if self.embed_scale is not None:
            x *= self.embed_scale
            # print('embed_scale embedding:', tokens.shape, x.shape)
            # print(self.embed_scale, torch.max(x), torch.min(x))

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)
            # print('self.embed_positions(tokens, positions=positions)', (self.embed_positions(tokens, positions=positions)).shape, x.shape)
            # print(torch.max(x), torch.min(x))
            # print('embed_positions:', tokens.shape, x.shape)
            # print('self.padding_idx:', self.padding_idx)
            # print('positions', positions)
            # print('src_lengths.shape',src_lengths.shape)
            # print('src_lengths', src_lengths)
            # print('self.vocab_size', self.vocab_size)

        # assert self.emb_layer_norm is None
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
            # print('emb_layer_norm:', tokens.shape, x.shape)
            # print(torch.max(x), torch.min(x))
        x = self.dropout_module(x)
        # print('dropout_module:', tokens.shape, x.shape)
        # print(torch.max(x), torch.min(x))


        # account for padding while computing the representation
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            # print('x.masked_fill:', tokens.shape, x.shape)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)
        # print('x:', tokens.shape, x.shape)
        key, value = self.pfcl(x)
        # print('c_temporal_k,v:', c_temporal_k.shape, c_temporal_v.shape)
        # print(torch.max(x), torch.min(x), torch.max(c_temporal_k), torch.min(c_temporal_v))

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
            # print('not last_state_only', tokens.shape, x.shape)

        for i in range(self.num_layers):
            if self.tie_layer_weights:
                j = 0
            else:
                j = i
            x, key, value, _ = self.layers[j](x=x, k=key, v=value)
            if not last_state_only:
                inner_states.append(x)

        # print('after stacked layers', x.shape)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            # print('padding_mask', x.shape)

        if self.sen_rep_type == 'mp':
            sentence_rep = x.sum(dim=-2) / src_lengths.unsqueeze(1)
            # print('final', torch.max(x), torch.min(x), torch.max(c_temporal_k), torch.min(c_temporal_v))
            # exit(0)
            # print('mp x.sum(dim=-2).shape', x.sum(dim=-2).shape)
            # print('mp src_lengths', src_lengths)
            # print('mp src_lengths.unsqueeze(1)', src_lengths.unsqueeze(1))
            # print('mp src_lengths.unsqueeze', src_lengths.shape)
            # print('mp src_lengths.unsqueeze(1)', src_lengths.unsqueeze(1).shape)
        else:
            sentence_rep = x[:, 0, :]
            # print('non mp sentence_rep', sentence_rep.shape)

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            # print('self.traceable', sentence_rep.shape)
            return torch.stack(inner_states), sentence_rep
        else:
            # print('non self.traceable', sentence_rep.shape)
            return inner_states, sentence_rep