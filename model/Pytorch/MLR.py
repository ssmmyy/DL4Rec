# -*- coding: utf-8 -*-
"""
The implementation of MLR: Mixed Logistic Regression
Created on Jan.3, 2020
@author: Mingyou Sun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLR(nn.Module):
    def __init__(self, num_m, num_features):
        """

        :param num_m: the number of MLR
        :param num_features: the number of features
        """

        super(MLR, self).__init__()
        self.num_m = num_m  # M

        # for clastering
        self.u = nn.Linear(num_features, num_m)  # F * M
        # for classing
        self.w = nn.Linear(num_features, num_m)  # F * M

    def forward(self, feature_value, label):
        """

        :param feature_value:
        :param label:
        :return:
        """

        U = self.u(feature_value)
        p1 = F.softmax(U)

        W = self.w(feature_value)
        p2 = F.sigmoid(W)

        y_output = torch.sum(p1 * p2, 1)

        return y_output
