import numpy as np
import torch
from torch import nn
from Discriminator import *
from Enconder import Encoder


class net(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, operator, dropout, alpha):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, output_dim, dropout, operator)
        self.discriminator = Discriminator(output_dim, alpha)

    def forward(self, user_list, intention_list, device):
        for user in user_list:
            data = user.origin_PYG_data.to(device)
            user.feature = self.encoder(data, device)
            user.aggr_feature = self.encoder(data, device, True)
            for intention in user.intention_list:
                intention.aggr_feature = self.encoder(intention.PYG_data().to(device), device, True)
                intention.feature = self.encoder(intention.PYG_data().to(device), device)

        for intention in intention_list:
            intention.aggr_feature = self.encoder(intention.PYG_data().to(device), device, True)
            intention.feature = self.encoder(intention.PYG_data().to(device), device)

        loss = self.discriminator(user_list, intention_list, device)
        return loss

    def get_infomax(self, user, intention):
        return self.discriminator.get_infomax(user, intention)
