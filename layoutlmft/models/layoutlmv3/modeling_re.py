import copy
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from transformers.file_utils import ModelOutput


@dataclass
class ReOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    entities: Optional[Dict] = None
    relations: Optional[Dict] = None
    pred_relations: Optional[Dict] = None


class BiaffineAttention(torch.nn.Module):
    """Implements a biaffine attention operator for binary relation classification.

    PyTorch implementation of the biaffine attention operator from "End-to-end neural relation
    extraction using deep biaffine attention" (https://arxiv.org/abs/1812.11275) which can be used
    as a classifier for binary relation classification.

    Args:
        in_features (int): The size of the feature dimension of the inputs.
        out_features (int): The size of the feature dimension of the output.

    Shape:
        - x_1: `(N, *, in_features)` where `N` is the batch dimension and `*` means any number of
          additional dimensisons.
        - x_2: `(N, *, in_features)`, where `N` is the batch dimension and `*` means any number of
          additional dimensions.
        - Output: `(N, *, out_features)`, where `N` is the batch dimension and `*` means any number
            of additional dimensions.

    Examples:
        >>> batch_size, in_features, out_features = 32, 100, 4
        >>> biaffine_attention = BiaffineAttention(in_features, out_features)
        >>> x_1 = torch.randn(batch_size, in_features)
        >>> x_2 = torch.randn(batch_size, in_features)
        >>> output = biaffine_attention(x_1, x_2)
        >>> print(output.size())
        torch.Size([32, 4])
    """

    def __init__(self, in_features, out_features):
        super(BiaffineAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.bilinear = torch.nn.Bilinear(in_features, in_features, out_features, bias=False)
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

class RELoss(nn.Module):
    def __init__(self):
        super(RELoss, self).__init__()
        # self.ce_weight  = CrossEntropyLoss(weight=torch.tensor([0.2, 2.0]))
        self.ce         = CrossEntropyLoss()

    def forward(self, output, target):
        loss = self.ce(output, target)
        return loss

class REDecoder(nn.Module):
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
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiaffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities):
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
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
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

    def get_predicted_relations(self, logits, relations, entities, index_offset=0):
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
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []

        batch_logits_all = []
        for b in range(batch_size):

            # entities_start_index保存每个实体的起始索引
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            batch_pred_result = []
            logits_all = []
            # add new : Matrix Dimension Limits
            num_pair = len(relations[b]["head"]) // 30000 + 1
            num_sample_batch = math.ceil(len(relations[b]["head"]) / num_pair)

            for index in range(0, len(relations[b]["head"]), num_sample_batch):
                # if index > 0: print(len(relations[b]["head"]))
                head_entities = torch.tensor(relations[b]["head"][index: index+num_sample_batch], device=device)
                tail_entities = torch.tensor(relations[b]["tail"][index: index+num_sample_batch], device=device)
                head_index = entities_start_index[head_entities]   # 获取head_entities实体在原文中的位起始索引
                head_label = entities_labels[head_entities]   # 获取head_entities实体所对应的label
                head_label_repr = self.entity_emb(head_label)

                tail_index = entities_start_index[tail_entities]
                tail_label = entities_labels[tail_entities]
                tail_label_repr = self.entity_emb(tail_label)

                head_repr = torch.cat(
                    (hidden_states[b][head_index], head_label_repr),
                    dim=-1,
                )
                tail_repr = torch.cat(
                    (hidden_states[b][tail_index], tail_label_repr),
                    dim=-1,
                )
                # print(head_repr.shape, tail_repr.shape)
                heads = self.ffnn_head(head_repr)
                tails = self.ffnn_tail(tail_repr)
                logits = self.rel_classifier(heads, tails)
                pred_relations = self.get_predicted_relations(logits, relations[b], entities[b], index_offset=index)
                logits_all.append(logits)
                batch_pred_result.extend(pred_relations)

            # loss
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            logits_all = torch.concat(logits_all, 0)
            # print(logits_all.shape, logits_all.device)
            loss += self.loss_fct(logits_all, relation_labels)
            # pred result
            all_pred_relations.append(batch_pred_result)
            batch_logits_all.append(logits_all.to("cpu"))
        batch_logits_all = torch.concat(batch_logits_all, 0)
        return loss, batch_logits_all, all_pred_relations