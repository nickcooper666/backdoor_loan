import torch
import torch.nn as nn
import torch.nn.functional as F
from models.simple import SimpleNet
import torch.nn.functional as F


class LoanNet(SimpleNet):
    def __init__(self, in_dim=91, n_hidden_1=46, n_hidden_2=23, out_dim=9, name=None, created_time=None):
        super(LoanNet, self).__init__(f'{name}_Simple', created_time)

        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x