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
    
        