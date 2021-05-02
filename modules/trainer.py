import numpy as np
import torch
import time
import pickle
import json
import os
from tqdm import tqdm

from modules.model import Model
from modules.optimizer import SGD
from modules.worker import Worker
from modules.client import Client
from utils.helper_funcs import Metrics
from modules.quantizer import encode, decode

class Trainer:
    def __init__(self, args, dataset):
        self.args = args

        self.all_train_data_size = 0
        self.bs = args['bs']
        self.num_round = args['num_round']
        self.clients_per_round = args['clients_per_round']
        self.verbose = args['verbose']
        
        self.subsampling = args['subsampling']
        self.gamma = args['subsampling_gamma']
        
        self.gpu = args['gpu']
        
        self.secure = args['secure']
        if self.secure:
            self.clip = args['secure_clip'] 
            self.epsilon = args['secure_epsilon']
            self.delta = args['secure_delta']
            

        self.model = Model(args)
        self.move_model_to_gpu(self.model, args)
        self.optimizer = SGD(self.model.parameters(), lr=args['lr'], weight_decay=args['wd'])
        self.worker = Worker(self.model, self.optimizer, args)
        self.clients = self.setup_clients(dataset)
        self.metrics = Metrics(self.clients, args)

        self.latest_model = self.worker.get_flat_model_params()

    @staticmethod
    def move_model_to_gpu(model, args):
        if 'gpu' in args and (args['gpu'] is True):
            device = 0 if 'device' not in args else args['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset):
        users, train_data, test_data = dataset

        all_clients = []
        for user in users:
            self.all_train_data_size += len(train_data[user])
            c = Client(user, train_data[user], test_data[user], self.bs, self.worker,
                       self.subsampling, self.gamma)
            all_clients.append(c)
        return all_clients

    def select_clients(self, seed=1):
        num_clients = self.clients_per_round
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in tqdm(range(self.num_round)):

            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            selected_clients = self.select_clients(seed=round_i)

            solns, stats = self.local_train(round_i, selected_clients)

            self.latest_model = self.aggregate(solns, stats=stats)
            if self.args['lr_sched']==1:
                self.optimizer.inverse_prop_decay_learning_rate(round_i, self.args['local_iters'])
            elif self.args['lr_sched']==2:
                self.optimizer.inverse_prop_decay_learning_rate2(round_i, self.args['local_iters'])

        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        self.metrics.write()

    def local_train(self, round_i, selected_clients, **kwargs):
        solns = []
        stats = []
        for i, c in enumerate(selected_clients, start=1):
            c.set_flat_model_params(self.latest_model)

            soln, stat = c.train_client()
            stat['sample_size'] = c.train_data.data.shape[0]
            if self.verbose:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Loss {:>.4f} | Acc {:>5.2f}%".format(
                    round_i, c.cid, i, self.clients_per_round,
                    stat['loss'], stat['acc'] * 100))

            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def aggregate(self, solns, **kwargs):
        aggregated_soln = torch.zeros_like(self.latest_model)

        num = 0
        for i, local_soln in enumerate(solns):
            if self.secure:
                #sigma_orig = np.sqrt(2 * np.log(1.25 / self.delta)) * self.clip / self.epsilon
                sigma_ampl = np.sqrt(8 * self.gamma ** 2 * np.log(1.25 * self.gamma / self.delta)) * self.clip / self.epsilon
                if self.gpu:
                    noise = torch.cuda.FloatTensor(local_soln.shape).normal_(0, sigma_ampl)
                else:
                    noise = torch.FloatTensor(local_soln.shape).normal_(0, sigma_ampl)
                noise /= kwargs['stats'][i]['sample_size']
                local_soln += noise
            if self.args['quantize']:
                code = encode(local_soln - self.latest_model, self.args['quan_level'])
                local_soln = decode(code, self.args['gpu']) + self.latest_model
            aggregated_soln += local_soln
            num += 1
        aggregated_soln /= num
        if self.args['verbose2']:
            print('sample_size: ~{}\t subsample_size: ~{}'.format(kwargs['stats'][i]['sample_size'], kwargs['stats'][i]['subsample_size']))
            if self.secure:
                print('sigma: {}\t nn: ~{}'.format(sigma_ampl, sigma_ampl/kwargs['stats'][i]['sample_size']))

        return aggregated_soln.detach()

    def test_latest_model_on_traindata(self, round_i):
        stats = self.local_test(use_eval_data=False)
        self.metrics.update_train_stats(round_i, stats)
        if self.verbose or self.args['verbose2']:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'.format(
                round_i, stats['acc'], stats['loss']))
            print('=' * 102 + "\n")

    def test_latest_model_on_evaldata(self, round_i):
        stats = self.local_test(use_eval_data=True)
        self.metrics.update_eval_stats(round_i, stats)
        if self.verbose:
            print('= Test = Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'.format(
                round_i, stats['acc'], stats['loss']))
            print('=' * 102 + "\n")


    def local_test(self, use_eval_data=True):
        self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for c in self.clients:
            tot_correct, num_sample, loss = c.test_client(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids}

        return stats


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


