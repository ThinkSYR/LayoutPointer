# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
from .modeling_gpdecoder import multilabel_categorical_crossentropy_v2, GlobalPointer
INF = 1e12


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed
    

class FFN(nn.Module):
    def __init__(self, config, output_hidden_size=None, is_pointer=False):
        super().__init__()
        self.config = config
        self.is_pointer = is_pointer
        output_hidden_size = config.hidden_size if output_hidden_size is None else output_hidden_size

        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, output_hidden_size)
        if not self.is_pointer:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.norm = RMSNorm(output_hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor=None):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.dense2(hidden_states)
        
        if not self.is_pointer:
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.norm(hidden_states + input_tensor)
        else:
            # hidden_states = self.norm(hidden_states)
            pass
        return hidden_states


class RotaryPositionEmbedding_2D(nn.Module):
    """二维RoPE
    https://spaces.ac.cn/archives/8397
    https://blog.eleuther.ai/rotary-embeddings/
    """
    def __init__(self, indices_x=None, indices_y=None, using_1d=False, head_size=64):
        super().__init__()
        self.indices_x = indices_x
        self.indices_y = indices_y
        base = 10000
        self.attn_dim = head_size
        self.using_1d = using_1d
        if self.using_1d:
            indices = torch.arange(0, self.attn_dim // 2, dtype=torch.float)
            self.indices_1d = nn.Parameter(torch.pow(base, -2 * indices / self.attn_dim), requires_grad=False)

    def sinusoidal_position_embedding(self, inputs, position_ids=None, indices_1d=None):
        """
        inputs: (bs, num_attention_heads, seq_len, head_size)
        """
        input_shape = inputs.size()
        batch_size, seq_len, output_dim = input_shape[0], input_shape[2], input_shape[3]
        
        repeat = False
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1).to(inputs.device)
            repeat = True
        if indices_1d is None:
            indices_1d = self.indices_1d.to(inputs.device) # 10000^(-2i/d)
        
        embeddings = position_ids * indices_1d
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) 
        if repeat:
            embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape)))) # batch_size repeat
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        
        cos_pos = embeddings[:, None, :, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = embeddings[:, None, :, ::2].repeat_interleave(2, dim=-1)
        
        return sin_pos, cos_pos

    def sinusoidal_position_embedding_2d(self, inputs, x, y, indices_x=None, indices_y=None):
        """
        和传统的sinusoidal_position_embedding不太一样，
        这里为了适应二维RoPE因此更改了indices
        inputs: (bs, num_attention_heads, seq_len, head_size)
        bbox: (batch_size, seq_len, 4) # (x1, y1, x2, y2)
        """
        input_shape = inputs.size()
        batch_size, seq_len, output_dim = input_shape[0], input_shape[2], input_shape[3]
        
        if indices_x is None:
            indices_x = self.indices_x
        if indices_y is None:
            indices_y = self.indices_y
        
        embeddings = torch.stack([indices_x*x, indices_y*y], dim=-1)
        embeddings = embeddings.reshape((batch_size, seq_len, output_dim // 2))
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        
        cos_pos = embeddings[:, None, :, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = embeddings[:, None, :, ::2].repeat_interleave(2, dim=-1)
        
        return sin_pos, cos_pos
    
    def apply_sin_cos_rope(self, x, sin_pos, cos_pos):
        x_2 = torch.stack([-x[..., 1::2], x[..., ::2]], -1)
        x_2 = x_2.reshape(x.shape)
        out = x * cos_pos + x_2 * sin_pos
        return out
    
    def forward(self, q, k, bbox):
        """
        q, k : (bs, num_attention_heads, seq_len, head_size)
        bbox : (batch_size, seq_len, 4) # (x1, y1, x2, y2)
        """
        if self.using_1d:
            sin_pos, cos_pos = self.sinusoidal_position_embedding(q)
            q = self.apply_sin_cos_rope(q, sin_pos, cos_pos)
            k = self.apply_sin_cos_rope(k, sin_pos, cos_pos)
            return q, k
        elif self.indices_x is None or self.indices_y is None:
            return q, k
        # 指定box的xy坐标
        x1 = bbox[..., 0].type(torch.float).unsqueeze(-1) 
        y1 = bbox[..., 1].type(torch.float).unsqueeze(-1)
        x2 = bbox[..., 2].type(torch.float).unsqueeze(-1)
        y2 = bbox[..., 3].type(torch.float).unsqueeze(-1)
        sin_pos_1, cos_pos_1 = self.sinusoidal_position_embedding_2d(q, x1, y1)
        sin_pos_2, cos_pos_2 = self.sinusoidal_position_embedding_2d(q, x2, y2)
        q[:, 0::2] = self.apply_sin_cos_rope(q[:, 0::2].clone(), sin_pos_1, cos_pos_1)
        q[:, 1::2] = self.apply_sin_cos_rope(q[:, 1::2].clone(), sin_pos_2, cos_pos_2)
        k[:, 0::2] = self.apply_sin_cos_rope(k[:, 0::2].clone(), sin_pos_1, cos_pos_1)
        k[:, 1::2] = self.apply_sin_cos_rope(k[:, 1::2].clone(), sin_pos_2, cos_pos_2)
        
        return q, k


class SCAAttention(nn.Module):
    def __init__(self, config, indices_x=None, indices_y=None, using_1d=False):
        super().__init__()
        self.config = config
        self.sca_add_slm = self.config.sca_add_slm
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.v_head_size = self.all_head_size

        # Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.v_head_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
            
        # RoPE-2D
        self.rope2d = RotaryPositionEmbedding_2D(head_size=self.attention_head_size, indices_x=indices_x, indices_y=indices_y, using_1d=using_1d)
        
        # output
        self.Output = nn.Linear(self.all_head_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        output_attentions=False,
        attention_bias=None,
        local_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        if bbox is not None:
            query_layer, key_layer = self.rope2d(
                query_layer, key_layer, bbox,
            )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The attention scores QT K/√d could be significantly larger than input elements, and result in overflow.
        # Changing the computational order into QT(K/√d) alleviates the problem. (https://arxiv.org/pdf/2105.13290.pdf)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # print(attention_scores.min(), attention_scores.max(), attention_scores.mean())

        if attention_bias is not None:
            attention_scores = attention_scores + attention_bias # (batch_size, heads, seq_len, seq_len)

        if attention_mask is not None:
            attention_mask = attention_mask[..., :512]
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, None, attention_scores.device)
            if self.sca_add_slm and local_attention_mask is not None:
                local_attn_mask = local_attention_mask
                extended_attention_mask = extended_attention_mask + local_attn_mask
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + extended_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output 
        context_layer = self.Output(context_layer)
        context_layer = self.out_dropout(context_layer)
        context_layer = self.norm(context_layer + hidden_states)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BaseModule(nn.Module):
    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) # 0.02
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SCALayer(BaseModule):
    def __init__(self, config, indices_x=None, indices_y=None, using_1d=False):
        super().__init__()
        self.config = config
        self.attention = SCAAttention(config, indices_x=indices_x, indices_y=indices_y, using_1d=using_1d)
        self.ffn = FFN(config)

    def forward(
        self, 
        hidden_states, 
        bbox, 
        attention_mask=None, 
        output_attentions=False,
        attention_bias=None,
        local_attention_mask=None,
    ):
        attention_outputs = self.attention(
            hidden_states,
            bbox,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            attention_bias=attention_bias,
            local_attention_mask=local_attention_mask,
        )
        self_outputs = attention_outputs[0]
        other_outputs = attention_outputs[1:]
        output = self.ffn(self_outputs, self_outputs)
        return output, other_outputs


class SCAPointer(BaseModule):
    def __init__(self, 
        config, 
        heads=4,
        tril_mask=True,
        indices_x=None,
        indices_y=None,
        using_1d=False,
    ):
        super().__init__()
        self.config = config
        self.heads = heads # config.num_labels 代表关系分类
        self.head_size = config.hidden_size // config.num_attention_heads
        self.tril_mask = tril_mask

        if self.config.pointer_with_sca:
            self.attention1 = SCAAttention(config, indices_x=indices_x, indices_y=indices_y, using_1d=using_1d)
            self.attention2 = SCAAttention(config, indices_x=indices_x, indices_y=indices_y, using_1d=using_1d)
        self.ffn1 = FFN(config, self.heads*self.head_size, is_pointer=True)
        self.ffn2 = FFN(config, self.heads*self.head_size, is_pointer=True)
        # self.ffn1 = nn.Linear(config.hidden_size, self.heads*self.head_size)
        # self.ffn2 = nn.Linear(config.hidden_size, self.heads*self.head_size)
        
        self.scale = 1 / (self.head_size**0.5) if self.config.pointer_scale else 1
        self.use_rope = self.config.pointer_rope
        if self.use_rope:
            self.ro = RotaryPositionEmbedding_2D(head_size=self.head_size, using_1d=True)
        
    def forward(
        self, 
        hidden_states,
        bbox, 
        attention_mask=None, 
        output_attentions=False,
        attention_bias=None,
        local_attention_mask=None,
    ):
        hidden_states_1 = hidden_states
        hidden_states_2 = hidden_states
            
        if self.config.pointer_with_sca:
            outputs1 = self.attention1(
                hidden_states_1, bbox, attention_mask=attention_mask,
                output_attentions=output_attentions, attention_bias=attention_bias,
                local_attention_mask=local_attention_mask,
            )
            output1, other_outputs1 = outputs1[0], outputs1[1:]
                
            outputs2 = self.attention2(
                hidden_states_2, bbox, attention_mask=attention_mask,
                output_attentions=output_attentions, attention_bias=attention_bias,
                local_attention_mask=local_attention_mask,
            )
            output2, other_outputs2 = outputs2[0], outputs2[1:]
        else:
            output1 = hidden_states_1
            output2 = hidden_states_2
        
        qw = self.ffn1(output1)
        kw = self.ffn2(output2)

        batch_size, seq_len, _ = qw.size()
        qw = qw.view(batch_size, seq_len, self.heads, self.head_size)
        kw = kw.view(batch_size, seq_len, self.heads, self.head_size)
        
        if self.use_rope:
            qw = qw.permute(0, 2, 1, 3)
            kw = kw.permute(0, 2, 1, 3)
            qw, kw = self.ro(qw, kw, None)
            qw = qw.permute(0, 2, 1, 3)
            kw = kw.permute(0, 2, 1, 3)
        
        # 计算内积
        # logits: (batch_size, heads, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw) * self.scale
        
        # mask
        batch_size, _, seq_len, seq_len = logits.size()
        if attention_mask.dim() == 2:
            pad_mask = attention_mask[..., :seq_len].unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
        else:
            pad_mask = attention_mask.unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * INF
            
        # 下三角排除
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * INF
        
        return logits


def get_local_attention_mask(bbox, device, attn_heads):
    """
    bbox : (batch_size, seq_len, 4)
    """
    # 计算中点坐标
    x_center = (bbox[:, :, 0] + bbox[:, :, 2]) / 2
    y_center = (bbox[:, :, 1] + bbox[:, :, 3]) / 2
    # 计算x, y的差值
    x_diff = x_center.unsqueeze(-1) - x_center.unsqueeze(-2)
    y_diff = y_center.unsqueeze(-1) - y_center.unsqueeze(-2)
    # x1, x2
    x1_diff = bbox[:, :, 0].unsqueeze(-1) - bbox[:, :, 0].unsqueeze(-2)
    x2_diff = bbox[:, :, 2].unsqueeze(-1) - bbox[:, :, 2].unsqueeze(-2)
    
    ## Soft
    x_min_diff = torch.min(torch.min(x1_diff.abs(), x2_diff.abs()), x_diff.abs())
    y_min_diff = y_diff.abs()
    x_diff = x_min_diff[:, None, :, :].repeat(1, attn_heads // 2, 1, 1)
    y_diff = y_min_diff[:, None, :, :].repeat(1, attn_heads // 2, 1, 1)
    attn_mask = torch.cat([x_diff, y_diff], dim=1).abs()
    
    return attn_mask.to(device)


def greedy_max_score_strategy(scores, threshold=0):
    """
    scores : (batch_size, class_num, seq_len, seq_len)
    """
    values, indices = scores.max(dim=2, keepdim=True)
    scores.fill_(threshold-1)
    scores.scatter_(2, indices, 1000.)
    return scores


class SCAREDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config = deepcopy(config)
        config.hidden_size += config.label_emb_size
        self.config = config
        
        if self.config.label_emb_size > 0:
            self.entity_emb = nn.Embedding(3, self.config.label_emb_size, scale_grad_by_freq=True)
        
        if self.config.sca_rope == 2:
            # theta for RoPE
            base = 10000
            self.attn_dim = config.hidden_size // config.num_attention_heads
            indices = torch.arange(0, self.attn_dim // 4, dtype=torch.float)
            self.indices_x = nn.Parameter(torch.pow(base, -4 * indices / self.attn_dim), requires_grad=True)
            self.indices_y = nn.Parameter(torch.pow(base, -4 * indices / self.attn_dim), requires_grad=True)
        else:
            self.indices_x = None
            self.indices_y = None
        using_1d = True if self.config.sca_rope == 1 else False
        
        # pre-layer and pointer
        self.link_heads = self.config.pointer_link_heads # 1
        self.decoder_layers = nn.ModuleList([
            SCALayer(config, indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d)
            for _ in range(self.config.pointer_sca_layer)
        ])
         
        # pointer choice
        self.with_gp = False
        if self.with_gp:
            self.s_o_head = GlobalPointer(
                config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
            self.s_o_tail = GlobalPointer(
                config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
        else:
            self.s_o_head = SCAPointer(
                config, heads=self.link_heads, tril_mask=False, 
                indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d,)
            self.s_o_tail = SCAPointer(
                config, heads=self.link_heads, tril_mask=False, 
                indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d,)
        
        if self.config.sca_add_slm:
            ## src
            # num_heads = self.config.num_attention_heads
            # sub1 = torch.tensor([1 / 2 ** h for h in range(num_heads//2)]).view(1, num_heads//2, 1, 1)
            # sub2 = sub1.clone()
            # self.alphax = nn.Parameter(torch.cat([sub1, sub2], dim=1), requires_grad=True)
            ## log
            self.gatex = nn.Linear(self.config.hidden_size, self.config.num_attention_heads)
            self.gamax = nn.Linear(self.config.hidden_size, self.config.num_attention_heads)
        # init
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.decoder_layers:
            layer.reset_parameters()
        self.s_o_head.reset_parameters()
        self.s_o_tail.reset_parameters()
         
        if self.config.label_emb_size > 0:
            self.entity_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.entity_emb.padding_idx is not None:
                self.entity_emb.weight.data[self.entity_emb.padding_idx].zero_()
        if self.config.sca_add_slm:
            self.gatex.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.gatex.bias is not None:
                self.gatex.bias.data.zero_()
            self.gamax.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.gamax.bias is not None:
                self.gamax.bias.data.zero_()

    def build_relation(self, relations, entities):
        """这部分构建关系,返回ground_truth：
        1. 从relations中找到所有的 relation_per_doc["head"] relation_per_doc["tail"] 代表两个实体对应的entities下标
        2. 从entities中找到所有的实体对应的下标，然后组合成所有的关系，生成大小为(batch_size, 1, seq_len, seq_len)的矩阵
        return:
            gt_head_link_index: (batch_size, heads, num_positives, 2) , 
            gt_tail_link_index: (batch_size, heads, num_positives, 2) , 
            batch_all_relations: (batch_size, num_relations, 4)
        """
        # 计算relation对齐
        batch_size = len(relations)
        # return 
        max_num_pos = 0
        gt_head_link_index_ = [[] for _ in range(batch_size)]
        gt_tail_link_index_ = [[] for _ in range(batch_size)]
        batch_all_relations = [[] for _ in range(batch_size)]
        attention_mask_ht = torch.zeros((batch_size, 512))
        for b in range(batch_size):
            # 实体太少了
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            # 实体坐标起止索引
            entities_start_index = torch.tensor(entities[b]["start"])
            entities_end_index   = torch.tensor(entities[b]["end"]) - 1 # 传入的标签数据的end是开区间的
            # 保留attn,dim=2
            attention_mask_ht[b, torch.cat([
                entities_start_index,
                entities_end_index
            ], dim=0)] = 1
            # 所有可能的关系
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            # 确保筛选，
            positive_relations = set([i for i in positive_relations if i in all_possible_relations]) 
            reordered_relations = list(positive_relations) + list(negative_relations)
            # 实体列表的下标
            heads_entind = torch.tensor([i[0] for i in reordered_relations])
            tails_entind = torch.tensor([i[1] for i in reordered_relations])
            # 实体真实起止位置
            from_ent_start_ind = entities_start_index[heads_entind]
            to_ent_start_ind   = entities_start_index[tails_entind]
            from_ent_end_ind   = entities_end_index[heads_entind]
            to_ent_end_ind     = entities_end_index[tails_entind]
            # 因为RE目前只有二类
            from_to_start_ind  = torch.stack([from_ent_start_ind, to_ent_start_ind], dim=1)
            from_to_end_ind    = torch.stack([from_ent_end_ind, to_ent_end_ind], dim=1)
            # 保存正例的用于计算loss
            gt_head_link_index_[b] = [from_to_start_ind[:len(positive_relations)]] # from_to_start_ind[len(positive_relations):], from_to_start_ind[:len(positive_relations)]]
            gt_tail_link_index_[b] = [from_to_end_ind[:len(positive_relations)]] # from_to_end_ind[len(positive_relations):], from_to_end_ind[:len(positive_relations)]]
            max_num_pos = max(max_num_pos, len(positive_relations), len(negative_relations))
            # 保存所有关系计算预测
            batch_all_relations[b] = torch.cat([from_to_start_ind, from_to_end_ind], dim=1)
        
        # return
        gt_head_link_index = torch.zeros((batch_size, self.link_heads, max_num_pos, 2), dtype=torch.long)
        gt_tail_link_index = torch.zeros((batch_size, self.link_heads, max_num_pos, 2), dtype=torch.long)
        for b in range(batch_size):
            for class_ in range(self.link_heads):
                gt_head_link_index[b, class_, :len(gt_head_link_index_[b][class_]), :] = gt_head_link_index_[b][class_]
                gt_tail_link_index[b, class_, :len(gt_tail_link_index_[b][class_]), :] = gt_tail_link_index_[b][class_]
            
        return gt_head_link_index, gt_tail_link_index, batch_all_relations, attention_mask_ht
            
    def get_predicted_relations(self, labels_all, batch_all_relations, so_head_outputs, so_tail_outputs, threshold=0):
        """
        so_head_outputs: (batch_size, class_num, seq_len, seq_len)
        """
        if self.config.pointer_greedy:
            # 贪心策略，选择列方向最大的值 key->value 选择最大的分数key
            so_head_outputs = greedy_max_score_strategy(so_head_outputs.detach(), threshold).cpu().numpy()
            so_tail_outputs = greedy_max_score_strategy(so_tail_outputs.detach(), threshold).cpu().numpy()
        else:
            so_head_outputs = so_head_outputs.detach().cpu().numpy()
            so_tail_outputs = so_tail_outputs.detach().cpu().numpy()
        pred_relations = []
        for b, each_relations in enumerate(batch_all_relations):
            batch_pred_result = []
            for sh, oh, st, ot in each_relations:
                p1s = np.where(so_head_outputs[b, :, sh, oh] > threshold)[0]
                p2s = np.where(so_tail_outputs[b, :, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    rel = {}
                    #rel["head_id"] = relations["head"][i]
                    rel["head"] = (sh.item(), st.item() + 1)
                    rel["head_type"] = labels_all[b, sh].item()

                    #rel["tail_id"] = relations["tail"][i]
                    rel["tail"] = (oh.item(), ot.item() + 1)
                    rel["tail_type"] = labels_all[b, oh].item()
                    
                    rel["type"] = 1
                    rel["score"] = "{:.6f}_{:.6f}".format(so_head_outputs[b, p, sh, oh], so_tail_outputs[b, p, st, ot])
                    
                    batch_pred_result.append(rel)
            pred_relations.append(batch_pred_result)
        return pred_relations

    def forward(self, hidden_states, entities, relations, attention_mask=None, bbox=None):
        if self.config.label_emb_size > 0:
            # label emb TODO: preprocess
            ent_labels = []
            for b in range(len(entities)):
                ent_label = torch.zeros((hidden_states.size(1),), dtype=torch.long)
                for i in range(len(entities[b]["start"])):
                    ent_label[entities[b]["start"][i]:entities[b]["end"][i]] = entities[b]["label"][i]
                ent_labels.append(ent_label)
            ent_labels = torch.stack(ent_labels, dim=0).to(hidden_states.device)
            ent_labels = self.entity_emb(ent_labels)
            hidden_states = torch.cat([hidden_states, ent_labels], dim=-1)
        
        if self.config.sca_add_slm:
            activate = lambda x: F.elu(x, alpha=1.0) + 1
            token_states = hidden_states
            alphax = activate(self.gatex(token_states)).unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
            gammax = activate(self.gamax(token_states)).unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
            local_attn_mask = get_local_attention_mask(bbox, hidden_states.device, self.config.num_attention_heads)
            alpha = - alphax * torch.log(1 + gammax * local_attn_mask)
        else:
            alpha = None

        # alpha = None
        attention_bias = None
        
        # decode
        output_attentions = ()
        for layer in self.decoder_layers:
            hidden_states, attention_probs = layer(
                hidden_states, bbox, attention_mask, 
                output_attentions=True, attention_bias=attention_bias, local_attention_mask=alpha,)
            output_attentions = output_attentions + attention_probs
        # relations matrix
        gt_head_link_index, gt_tail_link_index, batch_all_relations, _ = self.build_relation(relations, entities)
        gt_head_link_index = gt_head_link_index.to(hidden_states.device)
        gt_tail_link_index = gt_tail_link_index.to(hidden_states.device)
        # head and tail
        if self.with_gp:
            so_head_outputs = self.s_o_head(hidden_states, attention_mask)
            so_tail_outputs = self.s_o_tail(hidden_states, attention_mask)
        else:
            so_head_outputs = self.s_o_head(
                hidden_states, bbox, attention_mask, 
                attention_bias=attention_bias, local_attention_mask=alpha,)
            so_tail_outputs = self.s_o_tail(
                hidden_states, bbox, attention_mask, 
                attention_bias=attention_bias, local_attention_mask=alpha,)
        
        # compute loss
        loss = 0
        loss += multilabel_categorical_crossentropy_v2(gt_head_link_index, so_head_outputs, mask_zero=True)
        loss += multilabel_categorical_crossentropy_v2(gt_tail_link_index, so_tail_outputs, mask_zero=True)
        loss = loss / 2
        # predict relations and save 
        seq_len = hidden_states.size(1)
        # 构建一个平铺的连续存放的实体标签信息矩阵
        labels_all = torch.zeros((hidden_states.size(0), seq_len), dtype=torch.long)
        for b in range(hidden_states.size(0)):
            entities_start_index = torch.tensor(entities[b]["start"])
            entities_end_index   = torch.tensor(entities[b]["end"]) - 1
            entities_labels      = torch.tensor(entities[b]["label"])
            labels_all[b, entities_start_index] = entities_labels
            labels_all[b, entities_end_index]   = entities_labels
        
        pred_relations = self.get_predicted_relations(labels_all, batch_all_relations, so_head_outputs, so_tail_outputs)
        
        return loss, so_head_outputs, pred_relations, output_attentions


class SERPointer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config.pointer_rope = True
        
        if self.config.sca_rope == 2:
            # theta for RoPE
            base = 10000
            self.attn_dim = config.hidden_size // config.num_attention_heads
            indices = torch.arange(0, self.attn_dim // 4, dtype=torch.float)
            self.indices_x = nn.Parameter(torch.pow(base, -4 * indices / self.attn_dim), requires_grad=True)
            self.indices_y = nn.Parameter(torch.pow(base, -4 * indices / self.attn_dim), requires_grad=True)
        else:
            self.indices_x = None
            self.indices_y = None
        using_1d = True if self.config.sca_rope == 1 else False
        
        # pre-layer and pointer
        self.decoder_layers = nn.ModuleList([
            SCALayer(config, indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d)
            for _ in range(self.config.pointer_sca_layer)
        ])
        self.eht_pointer = SCAPointer(
            config, heads=self.config.pointer_cls_heads, tril_mask=True,
            indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d)
        
        if self.config.sca_add_slm:
            self.gatex = nn.Linear(self.config.hidden_size, self.config.num_attention_heads)
            self.gamax = nn.Linear(self.config.hidden_size, self.config.num_attention_heads)
        # init
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.decoder_layers:
            layer.reset_parameters()
        self.eht_pointer.reset_parameters()

    def forward(self, hidden_states, attention_mask=None, bbox=None):
        if self.config.sca_add_slm:
            ## log
            activate = lambda x: F.elu(x, alpha=1.0) + 1
            token_states = hidden_states
            alphax = activate(self.gatex(token_states)).unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
            gammax = activate(self.gamax(token_states)).unsqueeze(-1).permute(0, 2, 1, 3).contiguous()
            local_attn_mask = get_local_attention_mask(bbox, hidden_states.device, self.config.num_attention_heads)
            alpha = - alphax * torch.log(1 + gammax * local_attn_mask)
        else:
            alpha = None
        attention_bias = None
        
        # decode
        for layer in self.decoder_layers:
            hidden_states, _ = layer(
                hidden_states, bbox, attention_mask, 
                output_attentions=False, attention_bias=attention_bias,
                local_attention_mask=alpha,)
        # head and tail
        entity_outputs = self.eht_pointer(
            hidden_states, bbox, attention_mask, 
            attention_bias=attention_bias, local_attention_mask=alpha,)
        return entity_outputs

