# -*- coding: utf-8 -*-
"""
The implementation of DCN: Deep & Cross Network for Ad Click Predictions
Created on Dec.31, 2019
@author: Mingyou Sun
"""

import torch
import torch.nn as nn


class DCN(nn.Module):
    def __init__(self, embedding_dims, field_num, num_features, num_continuous_value, num_cross_layer,
                 deep_layers_output_num, dropout_deep):
        """

        :param embedding_dims: the dimension of feature embeddings.
        :param field_num: the number of fields.
        :param num_features: the number of all features.
        :param num_continuous_value: the number of continuous value per item.
        :param num_cross_layer: the number of cross layer.
        :param deep_layers_output_num: the output units of each layer.
        :param dropout_deep: the dropout rate of Deep part. The length is equal to the number of deep layer,
                                if not, extend to the this number.
        """
        super(DCN, self).__init__()

        self.field_num = field_num  # F
        self.embedding_dims = embedding_dims  # K
        self.num_continuous_value = num_continuous_value
        self.total_num = self.num_continuous_value + self.field_num * self.embedding_dims

        # output hidden size of each deep layer.
        self.deep_layers_output_num = deep_layers_output_num

        # Embedding
        self.feature_embedding = nn.Embedding(num_features, embedding_dims)  # M * K

        # Cross Network part
        self.cross_layers = [nn.Linear(self.total_num, 1) for i in range(num_cross_layer)]
        # Deep layer part
        self.deep_layers = [nn.Linear(self.total_num, deep_layers_output_num[0])]
        self.deep_layers.extend([nn.Linear(deep_layers_output_num[i], deep_layers_output_num[i + 1]) for i in
                                 range(len(deep_layers_output_num) - 1)])

        # Dropout
        self.dropout_deep = [nn.Dropout(dropout_deep[i]) for i in range(len(dropout_deep))]
        if len(self.dropout_deep) < len(deep_layers_output_num):
            self.dropout_deep.append(self.dropout_deep[-1])

        # F + K + last_layer_output_num
        self.output_layer = nn.Linear(self.total_num + deep_layers_output_num[-1], 1)

        # init all parts to glorot normal
        for i in range(num_cross_layer):
            nn.init.xavier_normal_(self.cross_layers[i].weight)
        for i in range(len(self.deep_layers)):
            nn.init.xavier_normal_(self.deep_layers[i].weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, feature_index, feature_value, numeric_value, label):
        """
            :param feature_index: sparse feature index N * F, the F is field number.
            :param feature_value: sparse feature value N * F, if is a sparse feature, the value is equal to 1.
            :param numeric_value: continuous feature value.
            :param label: ground truth, in here is not used.
            :return:
                y_output: the output of DCN N * 1
        """

        embeddings = self.feature_embedding(feature_index)  # B * F * K
        feature_value = feature_value.view((-1, self.field_num, 1))
        embeddings = embeddings * feature_value
        # concat numeric value
        x_0 = torch.cat((numeric_value, embeddings.view((-1, self.field_num * self.embedding_dims))), 1)
        # Cross Network part

        y_cross = torch.cat((numeric_value, embeddings.view((-1, self.field_num * self.embedding_dims))), 1)
        for i in range(len(self.cross_layers)):
            y_cross = self.cross_layers[i](torch.bmm(x_0.unsqueeze(2), y_cross.unsqueeze(1))).squeeze(2) + y_cross

        # Deep part
        # B * [F * K]
        y_deep = embeddings.view((-1, self.field_num * self.embedding_dims))
        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        # concat and final output
        y_concat = torch.cat((y_cross, y_deep), 1)
        y_output = self.output_layer(y_concat)

        return y_output
