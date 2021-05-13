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

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        hid = 25
        self.layer1 = nn.Linear(args['input_shape'], hid)
        self.layer2 = nn.Linear(hid, args['num_class'])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(F.relu(x))
        return x
    
    
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5, stride=2)
        self.fc2 = nn.Linear(12 * 12 * 3, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc2(torch.flatten(x, start_dim=1)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class CNN_old(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1((x.view(x.shape[0], 1, 28, 28))))
        #x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,3*3*64 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
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
    
        