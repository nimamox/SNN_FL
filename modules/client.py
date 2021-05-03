from torch.utils.data import DataLoader

from torch.utils.data import Subset
import numpy as np

from utils.helper_funcs import IterableAdapter

from NeuralEncBenchmark.surrogate_encoder import encode_data
from NeuralEncBenchmark.sparse_data_generator import sparse_generator

ISI_external_cache = {}

class Client:
    def __init__(self, cid, train_data, test_data, bs, worker, subsampling=False, gamma=1.0, args=None):
        self.args = args
        self.cid = cid
        self.worker = worker

        self.train_data = train_data
        self.test_data = test_data
        
        self.subsampling = subsampling
        self.gamma = gamma
        self.bs = bs
        
        if args['model'] == 'snn':
            self.train_ed = encode_data(train_data.data, train_data.labels, 
                                     nb_units=train_data.data.shape[1], encoder_type="ISI_inverse", nb_steps=args['nb_steps'], 
                                     TMAX=args['nb_steps'], external_ISI_cache=ISI_external_cache, batch_size=bs)        
            #self.train_dataloader = sparse_generator(self.train_ed, shuffle=True)
            self.train_dataloader = IterableAdapter(lambda: sparse_generator(self.train_ed, shuffle=True))
            
            self.test_ed = encode_data(test_data.data, test_data.labels, 
                                     nb_units=test_data.data.shape[1], encoder_type="ISI_inverse", nb_steps=args['nb_steps'], 
                                     TMAX=args['nb_steps'], external_ISI_cache=ISI_external_cache, batch_size=bs)
            #self.test_dataloader = sparse_generator(self.test_ed, shuffle=False)
            self.test_dataloader = IterableAdapter(lambda: sparse_generator(self.test_ed, shuffle=False))
        else:
            self.train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
            self.test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)

    def get_model_params(self):
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def train_client(self, **kwargs):
        if self.args['model'] == 'snn':
            if self.subsampling:
                #train_dataloader = sparse_generator(self.train_ed, shuffle=True, subsampling_ratio=self.gamma)
                train_dataloader = IterableAdapter(lambda: sparse_generator(self.train_ed, shuffle=True, subsampling_ratio=self.gamma))
            else:
                train_dataloader = IterableAdapter(lambda: sparse_generator(self.train_ed, shuffle=True))
            #self.test_dataloader = sparse_generator(self.test_ed, shuffle=False)
        else:
            if self.subsampling:
                data_size = self.train_data.data.shape[0]
                ind = np.random.choice(range(data_size), int(data_size * self.gamma), replace=False)
                train_dataloader = DataLoader(Subset(self.train_data, ind), batch_size=self.bs, shuffle=True)
            else:
                train_dataloader = self.train_dataloader
        
        local_solution, worker_stats = self.worker.train(train_dataloader, **kwargs)
        return local_solution, worker_stats

    def test_client(self, use_eval_data=True):
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        tot_correct, loss = self.worker.test(dataloader)
        return tot_correct, len(dataset), loss
