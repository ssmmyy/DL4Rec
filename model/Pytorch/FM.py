# -*- coding: utf-8 -*-
"""
The implementation of FM:Factorization-Machine for CTR Prediction
Created on Jan.2, 2020
@author: Mingyou Sun
"""

import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, embedding_dims, field_num, num_features):
        """

        :param embedding_dims: the dimension of feature embeddings.
        :param field_num: the number of fields.
        :param num_features: the number of all features.
        """
        super(FM, self).__init__()
        self.field_num = field_num  # F
        self.embedding_dims = embedding_dims  # K

        # Embedding(second order weight)
        self.feature_embedding = nn.Embedding(num_features, embedding_dims)  # M * K

        # FM part
        self.bias = nn.Parameter(torch.rand((1,)))
        self.FM_first_order_weights = nn.Embedding(num_features, 1)  # M * 1

    def forward(self, feature_index, feature_value, label):
        """

        :param feature_index: feature index N * F, the F is field number.
        :param feature_value: feature value N * F, if is a sparse feature, the value is equal to 1.
        :param label: ground truth, in here is not used.
        :return:
            y_output: the output of FM N * 1
        """
        # B * F * K
        embeddings = self.feature_embedding(feature_index)
        feature_value = feature_value.view((-1, self.field_num, 1))
        embeddings = embeddings * feature_value

        # FM part
        # FM first order
        first_order_weights = self.FM_first_order_weights(feature_index)

        # B * F
        y_first = torch.sum(first_order_weights * feature_value, 2)

        # FM part
        # FM second oreder
        summed_feature_emb_square = torch.sum(embeddings, 1) ** 2
        square_feature_emb_sum = torch.sum(embeddings ** 2, 1)
        # B * K
        y_second = 0.5 * (summed_feature_emb_square - square_feature_emb_sum)
        y_second = self.dropout_fm[-1](y_second)

        # final output
        y_output = y_first + y_second + self.bias

        return y_output
