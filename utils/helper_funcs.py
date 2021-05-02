import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

__all__ = ['mkdir', 'read_data', 'Metrics', "MiniDataset"]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def read_data(train_data_dir, test_data_dir, key=None):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    clients = []
    groups = []
    train_data = {}
    test_data = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    if key is not None:
        train_files = list(filter(lambda x: str(key) in x, train_files))

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    for cid, v in train_data.items():
        train_data[cid] = MiniDataset(v['x'], v['y'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    if key is not None:
        test_files = list(filter(lambda x: str(key) in x, test_files))

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data.update(cdata['user_data'])

    for cid, v in test_data.items():
        test_data[cid] = MiniDataset(v['x'], v['y'])

    clients = list(sorted(train_data.keys()))

    return clients, train_data, test_data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
            self.data = self.data.astype("float32")
            self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


class Metrics:
    def __init__(self, clients, args):
        self.args = args
        num_rounds = args['num_round'] + 1

        self.loss_on_train_data = [0] * num_rounds
        self.acc_on_train_data = [0] * num_rounds
        self.loss_on_eval_data = [0] * num_rounds
        self.acc_on_eval_data = [0] * num_rounds

        self.result_path = mkdir(os.path.join('./result', self.args['dataset']))
        prefix = 'S{}_eps{:.4f}_delt{:.4f}__clip{}_{:.3f}__SS{}_Gamma{:.3f}_lrs{}_bs{}'.format(int(args['secure']), 
                 args['secure_epsilon'], args['secure_delta'], int(args['clipping']), args['secure_clip'], 
                 int(args['subsampling']), args['subsampling_gamma'], args['lr_sched'], args['bs'])
        if args['quantize'] == False:
            suffix = 'E{}_M{}_s{}_R{}'.format(args['local_iters'], args['clients_per_round'], 100, args['num_round'])
        else:
            suffix = 'E{}_M{}_s{}_R{}'.format(args['local_iters'], args['clients_per_round'], args['quan_level'], args['num_round'])
        self.exp_name = '{}__{}'.format(prefix, suffix)
        print()

    def update_train_stats(self, round_i, train_stats):
        self.loss_on_train_data[round_i] = train_stats['loss']
        self.acc_on_train_data[round_i] = train_stats['acc']

    def update_eval_stats(self, round_i, eval_stats):
        self.loss_on_eval_data[round_i] = eval_stats['loss']
        self.acc_on_eval_data[round_i] = eval_stats['acc']

    def write(self):
        metrics = dict()

        # String
        metrics['dataset'] = self.args['dataset']
        metrics['num_round'] = self.args['num_round']
        metrics['lr'] = self.args['lr']
        metrics['local_iters'] = self.args['local_iters']
        metrics['bs'] = self.args['bs']

        metrics['loss_on_train_data'] = self.loss_on_train_data
        metrics['acc_on_train_data'] = self.acc_on_train_data
        metrics['loss_on_eval_data'] = self.loss_on_eval_data
        metrics['acc_on_eval_data'] = self.acc_on_eval_data
        
        metrics['secure'] = self.args['secure']
        metrics['secure_epsilon'] = self.args['secure_epsilon']
        metrics['secure_delta'] = self.args['secure_delta']
        
        metrics['clipping'] = self.args['clipping']
        metrics['secure_clip'] = self.args['secure_clip'] 
        
        metrics['subsampling'] = self.args['subsampling'] 
        metrics['subsampling_gamma'] = self.args['subsampling_gamma']    

        mkdir(os.path.join(self.result_path, self.exp_name))
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(str(metrics), ouf)