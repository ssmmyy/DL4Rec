# -*- coding: utf-8 -*-
"""
The implementation of NFM: Neural Factorization Machines for Sparse Predictive Analytics
Created on Jan.2, 2020
@author: Mingyou Sun
"""

import torch
import torch.nn as nn


class NFM(nn.Module):
    def __init__(self, embedding_dims, field_num, num_features, deep_layers_output_num, dropout_deep):
        """

        :param embedding_dims: the dimension of feature embeddings.
        :param field_num: the number of fields.
        :param num_features: the number of all features.
        :param deep_layers_output_num: the output units of each layer.
        :param dropout_deep: the dropout rate of Deep part. The length is equal to the number of deep layer,
                                if not, extend to the this number.
        """
        super(NFM, self).__init__()
        self.field_num = field_num  # F
        self.embedding_dims = embedding_dims  # K
        # output hidden size of each deep layer.
        self.deep_layers_output_num = deep_layers_output_num

        # Embedding
        self.feature_embedding = nn.Embedding(num_features, embedding_dims)  # M * K

        # FM part
        self.bias = nn.Parameter(torch.rand((1,)))
        self.FM_first_order_weights = nn.Embedding(num_features, 1)  # M * 1

        # Deep layer part
        self.deep_layers = [nn.Linear(field_num * embedding_dims, deep_layers_output_num[0])]
        self.deep_layers.extend([nn.Linear(deep_layers_output_num[i], deep_layers_output_num[i + 1]) for i in
                                 range(len(deep_layers_output_num) - 1)])

        # F + K + last_layer_output_num
        self.output_layer = nn.Linear(deep_layers_output_num[-1], 1)

        # activation function
        self.activate = nn.ReLU()
        # Dropout
        self.dropout_deep = [nn.Dropout(dropout_deep[i]) for i in range(len(dropout_deep))]
        if len(self.dropout_deep) < len(deep_layers_output_num):
            self.dropout_deep.append(self.dropout_deep[-1])

    def forward(self, feature_index, feature_value, label):
        """

        :param feature_index: feature index N * F, the F is field number.
        :param feature_value: feature value N * F, if is a sparse feature, the value is equal to 1.
        :param label: ground truth, in here is not used.
        :return:
            y_output: the output of NFM N * 1
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

        # NFM part
        # FM second oreder
        summed_feature_emb_square = torch.sum(embeddings, 1) ** 2
        square_feature_emb_sum = torch.sum(embeddings ** 2, 1)
        # B * K
        y_second = 0.5 * (summed_feature_emb_square - square_feature_emb_sum)
        y_second = self.dropout_fm[-1](y_second)

        # Deep part
        # B * [F * K]
        y_deep = y_second
        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        y_deep = self.output_layer(y_deep)

        # final output
        y_output = y_first + y_deep + self.bias

        return y_output
