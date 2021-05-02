import torch.nn as nn
import torch.nn.functional as F
import importlib
import math

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.layer = nn.Linear(args['input_shape'], args['num_class'])

    def forward(self, x):
        logit = self.layer(x)
        return logit