from collections import defaultdict
import copy
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .modeling_re import BiaffineAttention


class MLREDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.s_o_head = nn.ModuleList([
            copy.deepcopy(projection),  # ffnn_head
            copy.deepcopy(projection),  # ffnn_tail
            BiaffineAttention(config.hidden_size // 2, 2)  # rel_classifier
        ])
        self.s_o_tail = nn.ModuleList([
            copy.deepcopy(projection),  # ffnn_head
            copy.deepcopy(projection),  # ffnn_tail
            BiaffineAttention(config.hidden_size // 2, 2)  # rel_classifier
        ])
        self.tail_to_head = nn.ModuleList([
            copy.deepcopy(projection),  # ffnn_head
            copy.deepcopy(projection),  # ffnn_tail
            BiaffineAttention(config.hidden_size // 2, 2)  # rel_classifier
        ])
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities, rel_type=2):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
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
                if rel in [rel_type]: # 2代表head_to_head, 3代表tail_to_tail, 可能有2有3
                    positive_relations.add((relations[b]["head"][i], relations[b]["tail"][i]))
                    positive_rel_to_types[(relations[b]["head"][i], relations[b]["tail"][i])].append(rel)
            # 负样本
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities
    
    def build_ml_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
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
            # 正样本，仅针对于link==2 or 3 key-value
            positive_relations = set()
            for i, rel in enumerate(relations[b]["type"]):
                if rel == 1:
                    positive_relations.add((relations[b]["head"][i], relations[b]["tail"][i]))
            # 负样本
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)
            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) + [0] * (
                len(reordered_relations) - len(positive_relations)
            )
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities, index_offset=0, rel_type=1):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            i += index_offset
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = rel_type
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        head_relations, entities = self.build_relation(relations, entities, rel_type=2)
        tail_relations, entities = self.build_relation(relations, entities, rel_type=3)
        ml_realtions, entities = self.build_ml_relation(relations, entities)
        loss = 0
        all_pred_relations = [[] for _ in range(batch_size)]
        batch_logits_all = [None for _ in range(batch_size)]

        for relations, rel_type in zip([head_relations, tail_relations, ml_realtions], [2, 3, 1]):

            for b in range(batch_size):

                # entities_start_index保存每个实体的起始索引
                entities_start_index = torch.tensor(entities[b]["start"], device=device)
                entities_end_index = torch.tensor(entities[b]["end"], device=device) - 1
                entities_labels = torch.tensor(entities[b]["label"], device=device)
                batch_pred_result = []
                logits_all = []
                # add new : Matrix Dimension Limits
                # 防止显存占用过大
                num_pair = len(relations[b]["head"]) // 3000 + 1
                num_sample_batch = math.ceil(len(relations[b]["head"]) / num_pair)

                for index in range(0, len(relations[b]["head"]), num_sample_batch):
                    # 两方
                    subject_entities = torch.tensor(relations[b]["head"][index: index+num_sample_batch], device=device)
                    object_entities = torch.tensor(relations[b]["tail"][index: index+num_sample_batch], device=device)
                    # subject
                    shead_index = entities_start_index[subject_entities]   # 获取head_entities实体在原文中的位起始索引
                    shead_label = entities_labels[subject_entities]   # 获取head_entities实体所对应的label
                    shead_label_repr = self.entity_emb(shead_label)
                    
                    stail_index = entities_end_index[subject_entities]
                    stail_label = entities_labels[subject_entities]
                    stail_label_repr = self.entity_emb(stail_label)

                    # object
                    ohead_index = entities_start_index[object_entities]
                    ohead_label = entities_labels[object_entities]
                    ohead_label_repr = self.entity_emb(ohead_label)
                    
                    otail_index = entities_end_index[object_entities]
                    otail_label = entities_labels[object_entities]
                    otail_label_repr = self.entity_emb(otail_label)

                    if rel_type == 2:
                        key_repr = torch.cat((hidden_states[b][shead_index], shead_label_repr), dim=-1)
                        value_repr = torch.cat((hidden_states[b][ohead_index], ohead_label_repr), dim=-1)
                        heads = self.s_o_head[0](key_repr)
                        tails = self.s_o_head[1](value_repr)
                        logits = self.s_o_head[2](heads, tails)
                    elif rel_type == 3:
                        key_repr = torch.cat((hidden_states[b][stail_index], stail_label_repr), dim=-1)
                        value_repr = torch.cat((hidden_states[b][otail_index], otail_label_repr), dim=-1)
                        heads = self.s_o_tail[0](key_repr)
                        tails = self.s_o_tail[1](value_repr)
                        logits = self.s_o_tail[2](heads, tails)
                    elif rel_type == 1:
                        key_repr = torch.cat((hidden_states[b][stail_index], stail_label_repr), dim=-1)
                        value_repr = torch.cat((hidden_states[b][ohead_index], ohead_label_repr), dim=-1)
                        heads = self.tail_to_head[0](key_repr)
                        tails = self.tail_to_head[1](value_repr)
                        logits = self.tail_to_head[2](heads, tails)
                    
                    pred_relations = self.get_predicted_relations(logits, relations[b], entities[b], index_offset=index, rel_type=rel_type)
                    logits_all.append(logits)
                    batch_pred_result.extend(pred_relations)

                # loss
                relation_labels = torch.tensor(relations[b]["label"], device=device)
                logits_all = torch.concat(logits_all, 0)
                loss += self.loss_fct(logits_all, relation_labels)
                # pred result
                all_pred_relations[b].extend(batch_pred_result)
                batch_logits_all[b] = logits_all.to("cpu")
            
        batch_logits_all = torch.concat(batch_logits_all, 0)
        return loss, batch_logits_all, all_pred_relations