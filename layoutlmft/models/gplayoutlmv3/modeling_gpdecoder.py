import copy
import math

import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from transformers.file_utils import ModelOutput
INF = 1e12

@dataclass
class ReOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None


def multilabel_categorical_crossentropy(y_true, y_pred):
    """
    说明：
        1. y_true和y_pred的shape一致，y_true的元素是0～1
           的数，表示当前类是目标类的概率；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 和
           https://kexue.fm/archives/9064 。
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * INF  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * INF)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([(y_pred_neg), zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


def multilabel_categorical_crossentropy_v2(y_true, y_pred, mask_zero=False):
    '''另一种多标签交叉熵损失的torch实现
    注意:
        y_true的含义不同了,y_true存放的是每个分类(heads)对应的正样本位置坐标(num_positives,2) 
        二维坐标会加在一起,上限为seq_len*seq_len
    y_true : (batch_size, heads, num_positives, 2)
    y_pred : (batch_size, heads, seq_len, seq_len)
    '''
    shape = y_pred.shape
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + INF
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([(y_pos_2), zeros], dim=-1)
    
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)

    y_pred = torch.scatter(y_pred, -1, y_true, y_pos_2 - INF)
    y_pred_neg = torch.cat([y_pred, zeros], dim=-1)
    
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    
    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss


def SER_MC_CE_loss(y_true, y_pred):
    """
    y_true:(batch_size, heads, seq_len, seq_len)
    y_pred:(batch_size, heads, seq_len, seq_len)
    """
    y_true[y_true < 0] = 0
    batch_size, heads = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * heads, -1)
    y_pred = y_pred.reshape(batch_size * heads, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self,
        output_dim,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(SinusoidalPositionEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs, base=10000):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        input_shape = inputs.size()
        batch_size, seq_len = input_shape[0], input_shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, self.output_dim // 2, dtype=torch.float)
        indices = torch.pow(base, -2 * indices / self.output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)        
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, self.output_dim))

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        return embeddings
    

class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """
    def __init__(
        self,
        config,
        heads=4,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
        **kwargs
    ):
        super(GlobalPointer, self).__init__(**kwargs)
        self.config = config
        self.heads = heads
        self.emb_hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.num_attention_heads
        self.RoPE = RoPE
        self.use_bias = use_bias
        self.tril_mask = tril_mask
        self.scale = 1 / (self.head_size**0.5) if config.pointer_scale else 1
        
        if self.RoPE:
            self.Sinusoidal_PE = SinusoidalPositionEmbedding(self.head_size, 'zero')
        else:
            self.Sinusoidal_PE = None
        self.build()
            
    def build(self):
        self.dense = nn.Linear(
            self.emb_hidden_size,
            self.head_size * self.heads * 2,
            bias=self.use_bias,
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.constant_(self.dense.bias, 0)
            
    # def reset_parameters(self):
    #     self.apply(self._init_weights)

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
      
    def apply_rotary_position_embeddings(self, pos_emb, qw, kw):            
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, inputs, attention_mask=None, bbox=None):
        # 输入变换
        # (batch_size, seq_len, hidden_size)
        inputs = self.dense(inputs) 
        # (batch_size, seq_len, hidden_size*heads*2)
        inputs = torch.split(inputs, self.head_size*2, dim=-1)
        # [(batch_size, seq_len, hidden_size*2)] * (heads)
        inputs = torch.stack(inputs, dim=-2)
        # [(batch_size, seq_len, heads, hidden_size*2)]
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # RoPE编码
        if self.RoPE:
            pos = self.Sinusoidal_PE(inputs).to(qw.device)
            qw, kw = self.apply_rotary_position_embeddings(pos, qw, kw)
            
        # 计算内积
        # logits: (batch_size, heads, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw) * self.scale
        
        # mask
        batch_size, _, seq_len, seq_len = logits.size()
        pad_mask = attention_mask[..., :seq_len].unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * INF
            
        # 下三角排除
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * INF
        
        return logits


class GPREDecoder(nn.Module):
    def __init__(self, config, joint_re=False):
        super().__init__()
        from copy import deepcopy
        config = deepcopy(config)
        config.hidden_size += config.label_emb_size
        self.config = config
        if self.config.label_emb_size > 0:
            self.entity_emb = nn.Embedding(3, self.config.label_emb_size, scale_grad_by_freq=True)
        if joint_re:
            self.entity_output = GlobalPointer(config, heads=2, RoPE=True, use_bias=True, tril_mask=True)
        else:
            self.entity_output = None
        self.link_heads = 1
        self.s_o_head = GlobalPointer(config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
        self.s_o_tail = GlobalPointer(config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.s_o_head.reset_parameters()
        self.s_o_tail.reset_parameters()
        
        self.entity_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.entity_emb.padding_idx is not None:
            self.entity_emb.weight.data[self.entity_emb.padding_idx].zero_()

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
        for b in range(batch_size):
            # 实体太少了
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            # 实体坐标起止索引
            entities_start_index = torch.tensor(entities[b]["start"])
            entities_end_index   = torch.tensor(entities[b]["end"]) - 1 # 传入的标签数据的end是开区间的
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
            
        return gt_head_link_index, gt_tail_link_index, batch_all_relations
            

    def get_predicted_relations(self, labels_all, batch_all_relations, so_head_outputs, so_tail_outputs, threshold=0):
        """
        so_head_outputs: (batch_size, class_num, seq_len, seq_len)
        """
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
            
        so_head_outputs = self.s_o_head(hidden_states, attention_mask)
        so_tail_outputs = self.s_o_tail(hidden_states, attention_mask)
        # relations matrix
        gt_head_link_index, gt_tail_link_index, batch_all_relations = self.build_relation(relations, entities)
        gt_head_link_index = gt_head_link_index.to(hidden_states.device)
        gt_tail_link_index = gt_tail_link_index.to(hidden_states.device)
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
        
        return loss, so_head_outputs, pred_relations


class GPMultiLineRecognitionREDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        from copy import deepcopy
        config = deepcopy(config)
        config.hidden_size += config.label_emb_size
        self.config = config
        if self.config.label_emb_size > 0:
            self.entity_emb = nn.Embedding(3, self.config.label_emb_size, scale_grad_by_freq=True)
        self.link_heads = 1
        self.s_o_head = GlobalPointer(config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
        self.s_o_tail = GlobalPointer(config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
        self.tail_to_head = GlobalPointer(config, heads=self.link_heads, RoPE=False, use_bias=True, tril_mask=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.s_o_head.reset_parameters()
        self.s_o_tail.reset_parameters()
        self.tail_to_head.reset_parameters()
        
        self.entity_emb.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if self.entity_emb.padding_idx is not None:
            self.entity_emb.weight.data[self.entity_emb.padding_idx].zero_()

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
        for b in range(batch_size):
            # 实体太少了
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            # 实体坐标起止索引
            entities_start_index = torch.tensor(entities[b]["start"])
            entities_end_index   = torch.tensor(entities[b]["end"]) - 1 # 传入的标签数据的end是开区间的
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
            # 正样本，仅针对于link==2 or 3 key-value
            positive_relations = set()
            positive_rel_to_types = defaultdict(list)
            for i, rel in enumerate(relations[b]["type"]):
                if rel in [2, 3]: # 2代表head_to_head, 3代表tail_to_tail, 可能有2有3
                    positive_relations.add((relations[b]["head"][i], relations[b]["tail"][i]))
                    positive_rel_to_types[(relations[b]["head"][i], relations[b]["tail"][i])].append(rel)
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
            # gt_head_link_index_[b] = [from_to_start_ind[:len(positive_relations)]] # from_to_start_ind[len(positive_relations):], from_to_start_ind[:len(positive_relations)]]
            # gt_tail_link_index_[b] = [from_to_end_ind[:len(positive_relations)]] # from_to_end_ind[len(positive_relations):], from_to_end_ind[:len(positive_relations)]]
            gt_head_link_index_[b] = [[]]
            gt_tail_link_index_[b] = [[]]
            positive_relations = list(positive_relations)
            for i in range(len(positive_relations)):
                pos_rel = positive_relations[i]
                rel_types = positive_rel_to_types[pos_rel]
                for rel_type in rel_types:
                    if rel_type == 2: # 2代表head_to_head, 3代表tail_to_tail
                        gt_head_link_index_[b][0].append(from_to_start_ind[i])
                    elif rel_type == 3:
                        gt_tail_link_index_[b][0].append(from_to_end_ind[i])
            if len(gt_head_link_index_[b][0]) > 0:
                gt_head_link_index_[b][0] = torch.stack(gt_head_link_index_[b][0], dim=0)
            else:
                gt_head_link_index_[b][0] = torch.tensor([[0, 0]])
            if len(gt_tail_link_index_[b][0]) > 0:
                gt_tail_link_index_[b][0] = torch.stack(gt_tail_link_index_[b][0], dim=0)
            else:
                gt_tail_link_index_[b][0] = torch.tensor([[0, 0]])
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
            
        return gt_head_link_index, gt_tail_link_index, batch_all_relations
    
    def build_ml_relation(self, relations, entities):
        """这部分构建关系,返回ground_truth
        """
        # 计算relation对齐
        batch_size = len(relations)
        # return 
        max_num_pos = 0
        gt_tail_to_head_link_index_ = [[] for _ in range(batch_size)]
        batch_all_relations = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            # 实体太少了
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            # 实体坐标起止索引
            entities_start_index = torch.tensor(entities[b]["start"])
            entities_end_index   = torch.tensor(entities[b]["end"]) - 1 # 传入的标签数据的end是开区间的
            # 所有可能的关系
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if (entities[b]["label"][i] == entities[b]["label"][j]) and (i != j)
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            # 正样本，仅针对link==1
            positive_relations = set()
            for i, rel in enumerate(relations[b]["type"]):
                if rel == 1:
                    positive_relations.add((relations[b]["head"][i], relations[b]["tail"][i]))
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
            # 因为RE目前只有二类 tail to head
            from_end_to_start_ind  = torch.stack([from_ent_end_ind, to_ent_start_ind], dim=1)
            # 保存正例的用于计算loss
            gt_tail_to_head_link_index_[b] = [from_end_to_start_ind[:len(positive_relations)]] 
            max_num_pos = max(max_num_pos, len(positive_relations), len(negative_relations))
            # 保存所有关系计算预测
            batch_all_relations[b] = torch.stack([from_ent_start_ind, to_ent_start_ind, from_ent_end_ind, to_ent_end_ind], dim=1)
        
        # return
        gt_tail_to_head_link_index = torch.zeros((batch_size, self.link_heads, max_num_pos, 2), dtype=torch.long)
        for b in range(batch_size):
            for class_ in range(self.link_heads):
                gt_tail_to_head_link_index[b, class_, :len(gt_tail_to_head_link_index_[b][class_]), :] = gt_tail_to_head_link_index_[b][class_]
            
        return gt_tail_to_head_link_index, batch_all_relations
            
    def get_predicted_relations(self, labels_all, batch_all_relations, so_head_outputs, so_tail_outputs, threshold=0):
        """
        so_head_outputs: (batch_size, class_num, seq_len, seq_len)
        pred_relations: List[List[Dict num_predict] batch_size]
        """
        so_head_outputs = so_head_outputs.detach().cpu().numpy()
        so_tail_outputs = so_tail_outputs.detach().cpu().numpy()
        pred_relations = []
        for b, each_relations in enumerate(batch_all_relations):
            if len(each_relations) < 1:
                continue
            each_relations = each_relations.cpu().numpy()
            batch_pred_result = []
            
            sh_indices = each_relations[:, 0]
            oh_indices = each_relations[:, 1]
            st_indices = each_relations[:, 2]
            ot_indices = each_relations[:, 3]
            
            p1s = zip(*np.where(so_head_outputs[b, :, sh_indices, oh_indices] > threshold))
            p2s = zip(*np.where(so_tail_outputs[b, :, st_indices, ot_indices] > threshold))
            for p_type in [2, 3]:
                if p_type == 2:
                    ps = set(p1s)
                else: # if p_type == 3:
                    ps = set(p2s)
                for ind, p in ps:
                    sh, oh = sh_indices[ind], oh_indices[ind]
                    st, ot = st_indices[ind], ot_indices[ind]
                    rel = {}
                    #rel["head_id"] = relations["head"][i]
                    rel["head"] = (sh, st + 1)
                    rel["head_type"] = labels_all[b, sh].item()

                    #rel["tail_id"] = relations["tail"][i]
                    rel["tail"] = (oh, ot + 1)
                    rel["tail_type"] = labels_all[b, oh].item()
                    
                    rel["type"] = p_type
                    
                    batch_pred_result.append(rel)
            pred_relations.append(batch_pred_result)
        return pred_relations
     
    def get_predicted_ml_relations(self, labels_all, batch_all_relations, tail_to_head_outputs, threshold=0):
        """
        so_head_outputs: (batch_size, class_num, seq_len, seq_len)
        pred_relations: List[List[Dict num_predict] batch_size]
        """
        tail_to_head_outputs[..., 0] = 0
        tail_to_head_outputs = tail_to_head_outputs.argmax(dim=-1)
        tail_to_head_outputs = tail_to_head_outputs.detach().cpu().numpy()
        pred_relations = []
        for b, each_relations in enumerate(batch_all_relations):
            if len(each_relations) < 1:
                continue
            each_relations = each_relations.cpu().numpy()
            batch_pred_result = []
            
            sh_indices = each_relations[:, 0]
            oh_indices = each_relations[:, 1]
            st_indices = each_relations[:, 2]
            ot_indices = each_relations[:, 3]
            
            # p1s = zip(*np.where(tail_to_head_outputs[b, :, st_indices, oh_indices] > threshold))
            p1s = zip(*np.where(tail_to_head_outputs[b, 0, st_indices] == oh_indices))
            ps = set(p1s)
            
            for ind, in ps:
                sh, oh = sh_indices[ind], oh_indices[ind]
                st, ot = st_indices[ind], ot_indices[ind]
                rel = {}
                #rel["head_id"] = relations["head"][i]
                rel["head"] = (sh, st + 1)
                rel["head_type"] = labels_all[b, sh].item()

                #rel["tail_id"] = relations["tail"][i]
                rel["tail"] = (oh, ot + 1)
                rel["tail_type"] = labels_all[b, oh].item()
                
                rel["type"] = 1
                
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
            
        so_head_outputs = self.s_o_head(hidden_states, attention_mask)
        so_tail_outputs = self.s_o_tail(hidden_states, attention_mask)
        tail_to_head_outputs = self.tail_to_head(hidden_states, attention_mask)
        # relations matrix
        gt_head_link_index, gt_tail_link_index, batch_all_relations = self.build_relation(relations, entities)
        gt_tail_to_head_link_index, batch_ml_all_relations = self.build_ml_relation(relations, entities)
        gt_head_link_index = gt_head_link_index.to(hidden_states.device)
        gt_tail_link_index = gt_tail_link_index.to(hidden_states.device)
        gt_tail_to_head_link_index = gt_tail_to_head_link_index.to(hidden_states.device)
        # compute loss
        loss = 0
        loss += multilabel_categorical_crossentropy_v2(gt_head_link_index, so_head_outputs, mask_zero=True)
        loss += multilabel_categorical_crossentropy_v2(gt_tail_link_index, so_tail_outputs, mask_zero=True)
        loss = loss / 2
        loss += multilabel_categorical_crossentropy_v2(gt_tail_to_head_link_index, tail_to_head_outputs, mask_zero=True)
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
        
        # 推理，按照格式输出
        pred_relations = self.get_predicted_relations(labels_all, batch_all_relations, so_head_outputs, so_tail_outputs)
        pred_ml_relations = self.get_predicted_ml_relations(labels_all, batch_ml_all_relations, tail_to_head_outputs)
        for i in range(len(pred_relations)):
            pred_relations[i] += pred_ml_relations[i]
        
        return loss, so_head_outputs, pred_relations


