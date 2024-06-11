# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import defaultdict
from .modeling_gpdecoder import multilabel_categorical_crossentropy_v2, GPMultiLineRecognitionREDecoder
from .modeling_lpdecoder import SCAPointer, SCALayer, get_local_attention_mask
INF = 1e12


class SCAMLREDecoder(nn.Module):
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
            SCALayer(config, indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d), 
            SCALayer(config, indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d),
        ]) 
        self.s_o_head = SCAPointer(
            config, heads=self.link_heads, tril_mask=False, 
            indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d,)
        self.s_o_tail = SCAPointer(
            config, heads=self.link_heads, tril_mask=False, 
            indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d,)
        self.tail_to_head = SCAPointer(
            config, heads=self.link_heads, tril_mask=False, 
            indices_x=self.indices_x, indices_y=self.indices_y, using_1d=using_1d,)
        
        if self.config.sca_add_slm:
            self.gatex = nn.Linear(self.config.hidden_size, self.config.num_attention_heads)
            self.gamax = nn.Linear(self.config.hidden_size, self.config.num_attention_heads)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.decoder_layers:
            layer.reset_parameters()
        self.s_o_head.reset_parameters()
        self.s_o_tail.reset_parameters()
        self.tail_to_head.reset_parameters()
        
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

    def forward(self, hidden_states, entities, relations, attention_mask=None, bbox=None):
        if self.config.label_emb_size > 0:
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
            ## log
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
        for layer in self.decoder_layers:
            hidden_states, _ = layer(
                hidden_states, bbox, attention_mask, 
                output_attentions=False, attention_bias=attention_bias, local_attention_mask=alpha,)
        # relations matrix
        gt_head_link_index, gt_tail_link_index, batch_all_relations = self.build_relation(relations, entities)
        gt_tail_to_head_link_index, batch_ml_all_relations = self.build_ml_relation(relations, entities)
        gt_head_link_index = gt_head_link_index.to(hidden_states.device)
        gt_tail_link_index = gt_tail_link_index.to(hidden_states.device)
        gt_tail_to_head_link_index = gt_tail_to_head_link_index.to(hidden_states.device)
        # scores
        so_head_outputs = self.s_o_head(
            hidden_states, bbox, attention_mask, 
            attention_bias=attention_bias, local_attention_mask=alpha,)
        so_tail_outputs = self.s_o_tail(
            hidden_states, bbox, attention_mask, 
            attention_bias=attention_bias, local_attention_mask=alpha,)
        tail_to_head_outputs = self.tail_to_head(
            hidden_states, bbox, attention_mask, 
            attention_bias=attention_bias, local_attention_mask=alpha,)
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

