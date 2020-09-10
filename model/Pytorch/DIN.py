# -*- coding: utf-8 -*-
"""
The implementation of DIN: Deep Interest Network for Click-Through Rate Prediction
Created on Jan.7, 2020
@author: Mingyou Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINBase(nn.Module):
    def __init__(self, user_num, item_num, category_num, category_list, embedding_dims=64,
                 padding_idx=0,
                 deep_layers_output_num=(200, 80),
                 dropout_deep=(0.5, 0.5)):
        """

        :param user_num: the number of users.
        :param item_num: the number of items/ads.
        :param category_num: the number of categories.
        :param category_list: the category for each item.
        :param embedding_dims: the dimension of embedding.
        :param padding_idx: the index of padding, default to 0.
        :param deep_layers_output_num: the output unit of each dense layer.
        :param dropout_deep: the dropout rate of each dense layer.
        """
        self.user_num = user_num
        self.item_num = item_num
        self.category_num = category_num
        self.embedding_dims = embedding_dims  # D
        self.padding_index = padding_idx

        # user lookup embedding matrix
        self.user_embedding = nn.Embedding(user_num, embedding_dims, padding_idx=padding_idx)  # U * D

        # item lookup embedding matrix
        self.item_embedding = nn.Embedding(item_num, embedding_dims // 2, padding_idx=padding_idx)  # N * D/2

        # category lookup embedding matrix
        self.category_embedding = nn.Embedding(item_num, embedding_dims // 2, padding_idx=padding_idx)  # C * D/2

        # show the category of each item/ad
        self.category_list = torch.tensor(category_list, dtype=torch.int)  # N * 1

        # Deep layer part
        self.user_layer = nn.Linear(embedding_dims, embedding_dims)
        self.deep_layers = [nn.Linear(2 * embedding_dims, deep_layers_output_num[0])]
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

        self.batch_norm = nn.BatchNorm1d(embedding_dims)

    def get_item_score(self, user, item, history_item):
        """
        :param user: the user id, B * 1
        :param item: the item id, B * 1
        :param history_item: the history ids that user interaction with, B * S
        :return:
        """
        # B * 1
        item_category = self.category_list[item]
        # the max length of history
        history_length = history_item.shape[1]
        # B * S
        history_mask = (history_item != 0)
        # B * S
        history_item_category = self.category_list[history_item]
        # B * D
        item_embedding = torch.cat([self.item_embedding(item), self.category_embedding(item_category)], dim=-1)
        # B * S * D
        history_item_embeddings = torch.cat(
            [self.item_embedding(history_item), self.category_embedding[history_item_category]], dim=-1)

        # Base part.
        # mask 0
        history_item_embeddings = history_item_embeddings * history_mask.expand_as(history_item_embeddings)
        # the average of history item embedding, B * 2D
        history_embedding = torch.sum(history_item_embeddings, dim=1) / history_length

        # user embedding and din embedding
        history_embedding = self.batch_norm(history_embedding)
        user_embedding = self.user_layer(history_embedding)
        din_embedding = torch.cat([user_embedding, item_embedding], dim=-1)
        y_deep = din_embedding

        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        # (B,)
        y_output = self.output_layer(y_deep).squeeze(-1)

        return y_output

    def forward(self, user, item, neg_item, history_item):
        """

        :param user: the user id, B * 1
        :param item: the item id, B * 1
        :param label: the label item id, B * 1
        :param history_item: the history ids that user interaction with, B * S
        :return:
        """
        y_output = self.get_item_score(user=user, item=item, history_item=history_item)
        y_output_neg = self.get_item_score(user=user, item=neg_item, history_item=history_item)
        # (B,)
        return y_output - y_output_neg

    def predict(self, user, history_item, predict_num=None):
        """

        :param user: the user id, B * 1
        :param history_item: the history ids that user interaction with, B * S
        :param predict_num: the predict number, if None, predict all.
        :return:
        """
        # the max length of history
        history_length = history_item.shape[1]
        # B * S
        history_mask = (history_item != 0)
        # B * S
        history_item_category = self.category_list[history_item]
        # B * S * D
        history_item_embeddings = torch.cat(
            [self.item_embedding(history_item), self.category_embedding[history_item_category]], dim=-1)

        # Base part.
        # mask 0
        history_item_embeddings = history_item_embeddings * history_mask.expand_as(history_item_embeddings)
        # the average of history item embedding, B * 2D
        history_embedding = torch.sum(history_item_embeddings, dim=1) / history_length

        # user embedding and din embedding
        history_embedding = self.batch_norm(history_embedding)
        history_embedding = self.user_layer(history_embedding)

        # all embedding, N * D
        all_item_embedding = torch.cat([self.item_embedding.weight[1:], self.category_embedding(self.category_list)],
                                       dim=-1)
        # predict_num * D
        if predict_num is not None:
            sub_item_embedding = all_item_embedding[:predict_num]
        else:
            sub_item_embedding = all_item_embedding
        # predict_num * D -> B * predict_num * D
        sub_item_embedding = sub_item_embedding.unsqueeze(0).repeat(history_item.shape[0], 1, 1)
        # B * D -> B * predict_num * D
        sub_history_embedding = history_embedding.unsqueeze(1).expand_as(sub_item_embedding)
        sub_user_embedding = sub_history_embedding

        # B * predict_num * 2D
        sub_din_embedding = torch.cat([sub_user_embedding, sub_item_embedding], dim=-1)
        y_deep = sub_din_embedding

        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        # B * predict_num * 1 -> B * predict_num
        y_output = self.output_layer(y_deep).squeeze(-1)

        return y_output


class DICE(nn.Module):
    def __init__(self, embedding_dims, axis=-1, epsilon=1e-9):
        super(DICE, self).__init__()
        self.axis = axis
        self.epsilon = epsilon
        self.embedding_dims = embedding_dims
        self.batch_normalization = torch.nn.BatchNorm1d(embedding_dims, eps=epsilon)
        self.alphas = nn.Parameter(torch.zeros((embedding_dims,)))
        self.use_learning_phase = True

    def forward(self, input, training=None):
        inputs_normalized = self.batch_normalization(input)
        x_p = torch.sigmoid(inputs_normalized)
        return self.alphas * (1.0 - x_p) * input + x_p * input


class Attention(nn.Module):

    def __init__(self, hidden_size, attention_layers_output_num, dropout_attention,
                 attention_activation="dice"):
        self.hidden_size = hidden_size
        self.attention_layers_output_num = attention_layers_output_num
        self.dropout_attention = dropout_attention

        # Attention layer part
        self.attention_layers = [nn.Linear(4 * hidden_size, attention_layers_output_num[0])]
        self.attention_layers.extend(
            [nn.Linear(attention_layers_output_num[i], attention_layers_output_num[i + 1]) for i in
             range(len(attention_layers_output_num) - 1)])
        self.attention_output_layer = nn.Linear(attention_layers_output_num[-1], 1)

        # last_layer_output_num
        self.output_layer = nn.Linear(attention_layers_output_num[-1], 1)
        if attention_activation == "dice":
            self.activate = DICE(hidden_size)
        elif attention_activation == "relu":
            self.activate = nn.ReLU()
        else:
            self.activate == nn.Sigmoid()

    def forward(self, queries, keys, attention_mask):
        """

        :param queries: B * H
        :param keys: B * S * H
        :param attention_mask : B * S  0 or 1 indicates the padding or truth index
        :return:
        """
        # B * H -> B * S * H
        queries = queries.unsqueeze(1).repeat((1, keys.shape[1], 1))

        # B * S * 4H
        din_all = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)

        y_attention = din_all
        for i in range(len(self.deep_layers)):
            y_attention = self.activate(self.attention_layers[i](y_attention))
            y_attention = self.dropout_attention[i](y_attention)

        # B * 1 * S
        y_outputs = self.attention_output_layer(y_attention).transpose((1, 2))
        # B * S -> B * 1 * S
        attention_mask = attention_mask.unsqueeze(1)
        # a little value for padding ,not zero
        paddings = torch.ones_like(y_outputs) * (-1e12)
        # B * 1 * S
        y_outputs = torch.where(attention_mask, y_outputs, paddings)
        # Scale the value
        y_outputs = y_outputs / (keys.shape[-1] ** 0.5)
        # Get attention weight
        y_outputs = F.softmax(y_outputs)
        # B * 1 * S dot B * S * H -> B * 1 * H -> B * H
        y_outputs = torch.bmm(y_outputs, keys).squeeze(1)
        return y_outputs


class DIN(nn.Module):
    def __init__(self, user_num, item_num, category_num, category_list, embedding_dims=64,
                 padding_idx=0,
                 deep_layers_output_num=(200, 80),
                 dropout_deep=(0.5, 0.5), deep_activation="relu", attention_layers_output_num=(80, 40),
                 dropout_attention=(0.5, 0.5),
                 attention_activation="dice"):
        """

        :param user_num: the number of users.
        :param item_num: the number of items/ads.
        :param category_num: the number of categories.
        :param category_list: the category for each item.
        :param embedding_dims: the dimension of embedding.
        :param padding_idx: the index of padding, default to 0.
        :param deep_layers_output_num: the output unit of each dense layer.
        :param dropout_deep: the dropout rate of each dense layer.
        :param deep_activation: the activation function of dense layer.
        :param attention_layers_output_num: the output unit of each attention layer.
        :param dropout_attention: the dropout rate of each attention layer.
        :param attention_activation: the activation function of attention layer.
        """
        self.user_num = user_num
        self.item_num = item_num
        self.category_num = category_num
        self.embedding_dims = embedding_dims  # D
        self.padding_index = padding_idx

        # user lookup embedding matrix
        self.user_embedding = nn.Embedding(user_num, embedding_dims, padding_idx=padding_idx)  # U * D

        # item lookup embedding matrix
        self.item_embedding = nn.Embedding(item_num, embedding_dims // 2, padding_idx=padding_idx)  # N * D/2

        # category lookup embedding matrix
        self.category_embedding = nn.Embedding(item_num, embedding_dims // 2, padding_idx=padding_idx)  # C * D/2

        # show the category of each item/ad
        self.category_list = torch.tensor(category_list, dtype=torch.int)  # N * 1

        # Attention Layer
        self.attention = Attention(hidden_size=embedding_dims, attention_layers_output_num=attention_layers_output_num,
                                   dropout_attention=dropout_attention, attention_activation=attention_activation)

        # Deep layer part
        self.user_layer = nn.Linear(embedding_dims, embedding_dims)
        self.deep_layers = [nn.Linear(2 * embedding_dims, deep_layers_output_num[0])]
        self.deep_layers.extend([nn.Linear(deep_layers_output_num[i], deep_layers_output_num[i + 1]) for i in
                                 range(len(deep_layers_output_num) - 1)])

        # ast_layer_output_num
        self.output_layer = nn.Linear(deep_layers_output_num[-1], 1)

        # activation function
        if deep_activation == "relu":
            self.activate = nn.ReLU()
        elif deep_activation == "dice":
            self.activate = DICE(embedding_dims)
        else:
            self.activate = nn.Sigmoid()

        # Dropout
        self.dropout_deep = [nn.Dropout(dropout_deep[i]) for i in range(len(dropout_deep))]
        if len(self.dropout_deep) < len(deep_layers_output_num):
            self.dropout_deep.append(self.dropout_deep[-1])

        self.batch_norm = nn.BatchNorm1d(embedding_dims)

    def get_item_score(self, user, item, history_item):
        """
        :param user: the user id, B * 1
        :param item: the item id, B * 1
        :param history_item: the history ids that user interaction with, B * S
        :return:
        """
        # B * 1
        item_category = self.category_list[item]
        # the max length of history
        history_length = history_item.shape[1]
        # B * S
        history_mask = (history_item != 0)
        # B * S
        history_item_category = self.category_list[history_item]
        # B * D
        item_embedding = torch.cat([self.item_embedding(item), self.category_embedding(item_category)], dim=-1)
        # B * S * D
        history_item_embeddings = torch.cat(
            [self.item_embedding(history_item), self.category_embedding[history_item_category]], dim=-1)

        # Base part.
        # the average of history item embedding, B * 2D
        history_embedding = self.attention(queries=item_embedding, keys=history_item_embeddings,
                                           key_length=history_item_embeddings.shape[1], attention_mask=history_mask)

        # user embedding and din embedding
        history_embedding = self.batch_norm(history_embedding)
        user_embedding = self.user_layer(history_embedding)
        din_embedding = torch.cat([user_embedding, item_embedding], dim=-1)
        y_deep = din_embedding

        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        # (B,)
        y_output = self.output_layer(y_deep).squeeze(-1)

        return y_output

    def forward(self, user, item, neg_item, history_item):
        """

        :param user: the user id, B * 1
        :param item: the item id, B * 1
        :param history_item: the history ids that user interaction with, B * S
        :return:
        """
        y_output = self.get_item_score(user=user, item=item, history_item=history_item)
        y_output_neg = self.get_item_score(user=user, item=neg_item, history_item=history_item)
        # (B,)
        return y_output - y_output_neg

    def predict(self, user, history_item, predict_num=None):
        """

        :param user: the user id, B * 1
        :param history_item: the history ids that user interaction with, B * S
        :param predict_num: the predict number, if None, predict all.
        :return:
        """
        # the max length of history
        history_length = history_item.shape[1]
        # B * S
        history_mask = (history_item != 0)
        # B * S
        history_item_category = self.category_list[history_item]
        # B * S * D
        history_item_embeddings = torch.cat(
            [self.item_embedding(history_item), self.category_embedding[history_item_category]], dim=-1)

        # Base part.
        # mask 0
        history_item_embeddings = history_item_embeddings * history_mask.expand_as(history_item_embeddings)
        # the average of history item embedding, B * 2D
        history_embedding = torch.sum(history_item_embeddings, dim=1) / history_length

        # user embedding and din embedding
        history_embedding = self.batch_norm(history_embedding)
        history_embedding = self.user_layer(history_embedding)

        # all embedding, N * D
        all_item_embedding = torch.cat([self.item_embedding.weight[1:], self.category_embedding(self.category_list)],
                                       dim=-1)
        # predict_num * D
        if predict_num is not None:
            sub_item_embedding = all_item_embedding[:predict_num]
        else:
            sub_item_embedding = all_item_embedding
        # predict_num * D -> B * predict_num * D
        sub_item_embedding = sub_item_embedding.unsqueeze(0).repeat(history_item.shape[0], 1, 1)
        # B * D -> B * predict_num * D
        sub_history_embedding = history_embedding.unsqueeze(1).expand_as(sub_item_embedding)
        sub_user_embedding = sub_history_embedding

        # B * predict_num * 2D
        sub_din_embedding = torch.cat([sub_user_embedding, sub_item_embedding], dim=-1)
        y_deep = sub_din_embedding

        for i in range(len(self.deep_layers)):
            y_deep = self.activate(self.deep_layers[i](y_deep))
            y_deep = self.dropout_deep[i](y_deep)

        # B * predict_num * 1 -> B * predict_num
        y_output = self.output_layer(y_deep).squeeze(-1)

        return y_output
