# -*- coding: utf-8 -*-
"""
The implementation of PNN: Product-based neural networks for user response prediction
Created on Jan.2, 2020
@author: Mingyou Sun
"""

import torch
import torch.nn as nn


class PNN(nn.Module):

    def __init__(self, embedding_dims, field_num, num_features, pnn_output_dim, is_inner_product,
                 deep_layers_output_num, dropout_deep, is_decomp=True):
        """

        :param embedding_dims:
        :param field_num:
        :param num_features:
        :param deep_layers_output_num:
        :param dropout_deep:
        """
        super(PNN, self).__init__()

        self.field_num = field_num  # F
        self.embedding_dims = embedding_dims  # K
        # output hidden size of each deep layer.
        self.deep_layers_output_num = deep_layers_output_num
        self.pnn_output_dim = pnn_output_dim  # D1
        # Embedding
        self.feature_embedding = nn.Embedding(num_features, embedding_dims)  # M * K

        # Linear part
        self.linear_layers = nn.Linear(field_num * embedding_dims, pnn_output_dim)
        # PNN layer part
        self.is_inner_product = is_inner_product
        self.is_decomp = is_decomp
        if is_inner_product:
            if is_decomp:
                # Theta
                self.theta = nn.Parameter(torch.rand((pnn_output_dim, num_features)))
            else:
                self.inner_layer = nn.Linear(field_num * field_num, pnn_output_dim)
        else:
            self.outer_layer = nn.Linear(embedding_dims * embedding_dims, pnn_output_dim)
        # Bias
        self.bias = nn.Parameter(torch.rand((pnn_output_dim,)))

        # Deep layer part, in this paper, the number of layers is 2.
        self.deep_layers = [nn.Linear(pnn_output_dim, deep_layers_output_num[0])]
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

        :param feature_index: feature index N * F, the F is field number, and N is the batch size.
        :param feature_value: feature value N * F, if is a sparse feature, the value is equal to 1.
        :param label: ground truth, in here is not used.
        :return:
            y_output: the output of PNN N * 1
        """

        # B * F * K
        embeddings = self.feature_embedding(feature_index)
        feature_value = feature_value.view((-1, self.field_num, 1))
        embeddings = embeddings * feature_value

        # Linear part
        # B * [F * K] -> B * D1
        l_z = self.linear_layers(embeddings.view(-1, self.field_num * self.embedding_dims))

        # Inner product part
        if self.is_inner_product:
            # use decomposition
            if self.is_decomp:
                l_p = []
                for i in range(self.pnn_output_dim):
                    # B * F * K -> B * K
                    delta_sum = torch.sum(self.theta[i].unsequeeze(-1).repeat(1, self.embedding_dims) * embeddings,
                                          dim=1)
                    # B * K -> B * 1
                    l_p.append(torch.norm(torch.sum(delta_sum * delta_sum, dim=-1), dim=1, keepdim=True))
                # B * D1
                l_p = torch.cat(l_p, -1)
            else:  # do not use decomposition
                p = torch.bmm(embeddings, embeddings.transpose(1, 2)).view((-1, self.field_num * self.field_num))
                l_p = self.inner_layer(p)

        else:  # Outer product part
            # use decomposition
            if self.is_decomp:
                # B * F * K -> B * K
                f_sum = torch.sum(embeddings, dim=1)
                # B * K * K
                p = torch.bmm(f_sum.unsqueeze(2), f_sum.unsqueeze(1))
                l_p = self.outer_layer(p)
            else:
                print("Too complexity and time consuming.")

        # L1 and L2 and output layer
        y_deep = self.activate(l_z + l_p + self.bias)
        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        y_output = self.output_layer(y_deep)

        return y_output
