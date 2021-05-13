import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import math

import NeuralEncBenchmark
from NeuralEncBenchmark.surrogate_model import run_snn
from NeuralEncBenchmark.surrogate_train import init_model, compute_classification_accuracy, train

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.layer = nn.Linear(args['input_shape'], args['num_class'])

    def forward(self, x):
        logit = self.layer(x)
        return logit
    
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x.reshape(x.shape[0], 1, 28, 28))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class ModelSNN():
    def __init__(self, args):
        self.nb_steps = args['nb_steps']
        self.params, self.alpha, self.beta = init_model(nb_inputs=784, nb_hidden=25, nb_outputs=10, time_step=self.nb_steps)
    def parameters(self):
        return self.params
    def run_snn_model(self, inp_spike):
        #run_snn(inputs, batch_size, nb_steps, params, alpha, beta)
        surr_output, _ = run_snn(inp_spike, inp_spike.shape[0], self.nb_steps, self.params, self.alpha, self.beta)
        return surr_output.sum(1)
    
        