# -*- coding: utf-8 -*-
"""
The implementation of some Bandit algorithms
Created on Jan.3, 2020
@author: Mingyou Sun
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Bandit(nn.Module):
    def __init__(self, num_arm, total_num_test, method="LinUCB", alpha=0.25, r1=0.6, r0=-16, feature_dims=6):
        super(Bandit, self).__init__()

        self.num_arm = num_arm
        self.total_num_test = total_num_test
        self.method = method
        # LinUCB part
        if method == "LinUCB":
            self.alpha = alpha
            self.r1 = r1
            self.r0 = r0
            self.feature_dims = feature_dims
            self.Aa = []
            self.ba = []
            self.AaI = []
            self.theta = []
            self.set_init_params()

        # True rewards of each bandit
        self.true_rewards = torch.Tensor(num_arm, ).uniform_(0, 1)
        # Estimated rewards of each bandit, init to 0
        self.estimated_rewards_ucb = torch.zeros_like(self.true_rewards, dtype=torch.float)
        # The number of exploration of each bandit.
        self.explore_count_ucb = torch.zeros_like(self.true_rewards, dtype=torch.float)
        # Total rewards
        self.total_rewards_ucb = 0

        # Estimated rewards of each bandit, init to 0
        self.estimated_rewards_linucb = torch.zeros_like(self.true_rewards, dtype=torch.float)
        # The number of exploration of each bandit.
        self.explore_count_linucb = torch.zeros_like(self.true_rewards, dtype=torch.float)
        # Total rewards
        self.total_rewards_linucb = 0

    # Set LinUCB init parameters
    def set_init_params(self):
        for i in range(self.num_arm):
            self.Aa.append(torch.tensor(np.identity(self.feature_dims), dtype=torch.float))
            self.ba.append(torch.zeros((self.feature_dims, 1), dtype=torch.float))
            self.AaI.append(torch.tensor(np.identity(self.feature_dims), dtype=torch.float))
            self.theta.append(torch.zeros((self.feature_dims, 1), dtype=torch.float))
        self.Aa = torch.stack(self.Aa)
        self.ba = torch.stack(self.ba)
        self.AaI = torch.stack(self.AaI)
        self.theta = torch.stack(self.theta)

    # Calculate the delta for UCB
    def calculate_delta(self, num_test):
        delta = (self.explore_count_ucb == 0).float() + torch.sqrt(
            2 * torch.log10(num_test) / (self.explore_count_ucb + 1e-8) * (self.explore_count_ucb != 0).float())
        return delta

    # UCB
    def UCB_recommend(self, num_test):
        upper_bound_probs = self.estimated_rewards_ucb + self.calculate_delta(num_test)
        item = torch.argmax(upper_bound_probs)
        return item

    def UCB(self, num_test, input):
        num_test = torch.tensor(num_test).float()
        item = self.UCB_recommend(num_test)
        reward = torch.tensor(np.random.binomial(n=1, p=self.true_rewards[item]))
        self.total_rewards_ucb = self.total_rewards_ucb + reward
        explore_count = self.explore_count_ucb[item]
        self.estimated_rewards_ucb[item] = (explore_count * self.estimated_rewards_ucb[item] + reward) / (
                explore_count + 1)
        self.explore_count_ucb[item] = explore_count + 1
        print("UCB: \n\tTrue:", self.true_rewards, "\n\tChoosen:", self.explore_count_ucb, "\n\tEstimated:",
              self.estimated_rewards_ucb,
              "\n\tTotal:", self.total_rewards_ucb, "\n\tMSE:",
              torch.sum(torch.pow(self.estimated_rewards_ucb - self.true_rewards, 2)), "\n")

    def LinUCB_recommend(self, user_features):
        user_features = torch.randn_like(user_features)
        x_a = user_features.unsqueeze(0).repeat(self.AaI.shape[0], 1, 1)
        x_aT = x_a.transpose(1, 2)
        x = torch.tensor(user_features, dtype=torch.float)
        xT = x.t()
        # to reduce the compute time, the theta is defined and updated in Update function
        print(self.theta.shape, x_a.shape)
        score1 = torch.bmm(self.theta.transpose(1, 2), x_a).squeeze()
        score2 = self.alpha * torch.sqrt(torch.bmm(torch.bmm(x_aT, self.AaI), x_a)).squeeze()
        print(score1 + score2)
        item = torch.argmax(score1 + score2)
        self.x = x
        self.xT = xT
        return item

    def LinUCB_update(self, item, reward):
        # finish tag
        if reward == -1:
            return
        if reward == 1:
            r = self.r1
        else:
            r = self.r0
        # print("aaa",self.Aa[item].shape,torch.mm(self.x, self.xT).shape)
        self.Aa[item] = self.Aa[item] + torch.mm(self.x, self.xT)
        self.ba[item] = self.ba[item] + r * self.x
        self.AaI[item] = torch.inverse(self.Aa[item])
        # theta is defined and updated in here
        self.theta[item] = torch.mm(self.AaI[item], self.ba[item])
        return

    def LinUCB(self, num_test, user_features):

        item = self.LinUCB_recommend(user_features)
        reward = torch.tensor(np.random.binomial(n=1, p=self.true_rewards[item]))
        self.LinUCB_update(item, reward)
        self.total_rewards_linucb = self.total_rewards_linucb + reward
        explore_count = self.explore_count_linucb[item]
        self.estimated_rewards_linucb[item] = (explore_count * self.estimated_rewards_linucb[item] + reward) / (
                explore_count + 1)
        self.explore_count_linucb[item] = explore_count + 1
        print("LinUCB:\n\tTrue:", self.true_rewards, "\n\tChoosen:", self.explore_count_linucb, "\n\tEstimated:",
              self.estimated_rewards_linucb,
              "\n\tTotal:", self.total_rewards_linucb, "\n\tMSE:",
              torch.sum(torch.pow(self.estimated_rewards_linucb - self.true_rewards, 2)), "\n\n")
        return

    def forward(self, input):
        user_features = torch.FloatTensor(input)
        for num_test in range(1, self.total_num_test + 1):
            self.UCB(num_test, user_features)
            self.LinUCB(num_test, user_features)
        return


if __name__ == '__main__':
    model = Bandit(10, 100,feature_dims=2)
    x = torch.Tensor(2, 1)
    model(x)
