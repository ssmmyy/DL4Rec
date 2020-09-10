# -*- coding: utf-8 -*-
"""
The implementation of AFM: Attentional Factorization Machines Learning the Weight of Feature Interactions via Attention Networks
Created on Jan.2, 2020
@author: Mingyou Sun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFM(nn.Module):
    def __init__(self, embedding_dims, field_num, num_features, attention_dims):
        """

        :param embedding_dims: the dimension of feature embeddings.
        :param field_num: the number of fields.
        :param num_features: the number of all features.
        """
        super(AFM, self).__init__()
        self.field_num = field_num  # F
        self.embedding_dims = embedding_dims  # K
        self.attention_dims = attention_dims  # T
        # Embedding
        self.feature_embedding = nn.Embedding(num_features, embedding_dims)  # M * K

        # FM part
        self.bias = nn.Parameter(torch.rand((1,)))
        self.FM_first_order_weights = nn.Embedding(num_features, 1)  # M * 1

        # Attention part
        # Attention map layer
        self.attention_map_layer = nn.Linear(embedding_dims, attention_dims)
        # Attention output layer
        self.attention_output_layer = nn.Linear(attention_dims, 1, bias=False)
        self.activate = nn.ReLU()

        # Deep final output
        self.output_layer = nn.Linear(embedding_dims, 1, bias=False)

    def forward(self, feature_index, feature_value, label):
        """

        :param feature_index: feature index B * F, the F is field number.
        :param feature_value: feature value B * F, if is a sparse feature, the value is equal to 1.
        :param label: ground truth, in here is not used.
        :return:
            y_output: the output of AFM N * 1
        """
        # B * F * K
        embeddings = self.feature_embedding(feature_index)
        feature_value = feature_value.view((-1, self.field_num, 1))
        embeddings = embeddings * feature_value

        # AFM part
        # FM first order
        first_order_weights = self.FM_first_order_weights(feature_index)

        # B * F
        y_first = torch.sum(first_order_weights * feature_value, 2)

        # AFM second oreder
        product_list = []
        for i in range(self.field_num - 1):
            for j in range(i + 1, self.field_num):
                product_list.append(embeddings[:, i] * embeddings[:, j])  # B * K
        element_wise_product = torch.stack(product_list, dim=1)  # B * (F*(F-1))/2 * K

        # Attention part
        energy = self.activate(self.attention_map_layer(element_wise_product))  # B * (F*(F-1))/2 * T
        energy = self.attention_output_layer(energy).squeeze()  # B * (F*(F-1))/2
        attention_weights = F.softmax(energy)  # B * (F*(F-1))/2
        # AFM second order
        element_wise_product = element_wise_product * attention_weights.unsqueeze(-1).expand_as(element_wise_product)
        y_second = torch.sum(element_wise_product, 1)  # B * K
        y_second = self.output_layer(y_second)

        # final output
        y_output = y_first + y_second + self.bias

        return y_output
