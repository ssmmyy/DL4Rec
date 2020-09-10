# -*- coding: utf-8 -*-
"""
The implementation of NCF: Neural Collaborative Filtering
Created on Jan.2, 2020
@author: Mingyou Sun
"""
import torch
import torch.nn as nn


# NeuMF
class NCF(nn.Module):
    def __init__(self, embedding_dims, num_users, num_items, deep_layers_output_num, dropout_deep):
        super(NCF, self).__init__()
        # GMF part
        self.mf_user_embedding = nn.Embedding(num_users, embedding_dims)
        self.mf_item_embedding = nn.Embedding(num_items, embedding_dims)

        # Deep part
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dims)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dims)

        self.deep_layers = [nn.Linear(embedding_dims * 2, deep_layers_output_num[0])]
        self.deep_layers.extend([nn.Linear(deep_layers_output_num[i], deep_layers_output_num[i + 1]) for i in
                                 range(len(deep_layers_output_num) - 1)])

        # activation function
        self.activate = nn.ReLU()
        # Dropout
        self.dropout_deep = [nn.Dropout(dropout_deep[i]) for i in range(len(dropout_deep))]
        if len(self.dropout_deep) < len(deep_layers_output_num):
            self.dropout_deep.append(self.dropout_deep[-1])

        # F + K + last_layer_output_num
        self.output_layer = nn.Linear(embedding_dims + deep_layers_output_num[-1], 1)

    def forward(self, users, items, label):
        """

        :param users:
        :param items:
        :param label:
        :return:
        """
        # MF part
        mf_user_vector = self.mf_user_embedding(users)  # B * K
        mf_item_vector = self.mf_item_embedding(users)  # B * K
        y_gmf = mf_user_vector * mf_item_vector  # B * K

        # Deep part
        mlp_user_vector = self.mlp_user_embedding(users)  # B * K
        mlp_item_vector = self.mlp_item_embedding(items)  # B * K
        y_deep = torch.cat((mlp_user_vector, mlp_item_vector), dim=-1)
        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        y_output = torch.cat((y_gmf, y_deep), dim=-1)
        y_output = self.output_layer(y_output)

        return y_output
