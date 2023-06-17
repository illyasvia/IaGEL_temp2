import torch
import torch.nn.functional as F
from torch import nn

"""
local_discriminator: 鉴别用户节点视图和意图独立视图匹配程度
prior_discriminator
global_discriminator: 鉴别用户节点视图和意图重构维用户视图差异
"""


class record:
    def __init__(self, value, reward, punishment):
        self.value = value
        self.reward = reward
        self.punishment = punishment

    def __lt__(self, other):
        return self.value > other.value

    def __repr__(self):
        return "value={},reward={},punishment={}".format(self.value, self.reward, self.punishment)


class Discriminator(nn.Module):
    def __init__(self, input_dim, alpha):
        super().__init__()
        self.judge = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, input_dim),
            nn.ReLU(),
        )
        self.alpha = alpha
        self.BN = nn.BatchNorm1d(1)

    def get_local_loss(self, users_list, intention_list, device):
        explore_and_exploit = list()
        score = torch.tensor(0.0).to(device)
        for user in users_list:
            explore_and_exploit.clear()
            temp_score = torch.tensor(0.0).to(device)
            for intention in intention_list:
                pos = 0
                neg = 0
                for node in intention.node_list:
                    if node.id in user.item_list:
                        pos += 1
                    else:
                        neg += 1
                res = torch.matmul(self.judge(intention.aggr_feature), self.judge(user.feature).t()).max()
                res = F.softplus(res)
                explore_and_exploit.append(record(res, pos / (pos + neg), neg / (pos + neg)))
            for intention in user.intention_list:
                res = torch.matmul(self.judge(intention.aggr_feature), self.judge(user.feature).t()).max()
                res = F.softplus(res)
                explore_and_exploit.append(record(res, 1, 0))
            explore_and_exploit = sorted(explore_and_exploit)
            for x in explore_and_exploit[:20]:
                temp_score -= self.alpha * x.reward
                temp_score += (1 - self.alpha) * x.punishment
            print("temp_score:{}".format(temp_score))
        return score

    def get_global_loss(self, users_list, intention_list, device):
        sum_loss = None
        for user in users_list:
            pos = torch.tensor(0.0).to(device)
            neg = torch.tensor(0.0).to(device)
            cnt_pos = 0
            cnt_neg = 0
            for intention in intention_list:
                res = torch.matmul(self.judge(intention.aggr_feature), self.judge(user.feature).t()).max()
                res = F.softplus(res)
                pos += res
                cnt_pos += 1
            for intention in user.intention_list:
                res = torch.matmul(self.judge(intention.aggr_feature), self.judge(user.feature).t()).max()
                res = F.softplus(res)
                neg += res
                cnt_neg += 1
            loss = torch.unsqueeze(torch.unsqueeze(torch.tensor(neg / cnt_neg - pos / cnt_pos), 0), 0)
            print("pos={},neg={}".format(pos / cnt_pos, neg / cnt_neg))
            if sum_loss is None:
                sum_loss = torch.tensor(loss)
            else:
                sum_loss = torch.cat([sum_loss, loss])
        return self.BN(sum_loss).sum()

    def forward(self, users_list, intention_list, device):
        return self.get_local_loss(users_list, intention_list, device) + self.get_global_loss(users_list,
                                                                                              intention_list, device)

    def get_infomax(self, user, intention):
        return torch.matmul(self.judge(intention.aggr_feature), self.judge(user.feature).t()).max()
