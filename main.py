import numpy as np
import torch
import os
from modules.trainer import Trainer
from utils.helper_funcs import read_data
import json

def set_seed(seed, gpu=True):
    # Set random seed
    np.random.seed((1 + seed))
    torch.manual_seed(12 + seed)
    if gpu:
        torch.cuda.manual_seed_all(123 + seed)    

def main():
    args = dict()
    args['dataset'] = '2_digits_per_client'
    args['model'] = 'logistic'
    args['wd'] = 0.001
    args['verbose'] = False
    args['verbose2'] = True
    args['num_iters'] = 1000
    args['local_iters'] = 10
    args['num_round'] = args['num_iters'] // args['local_iters']
    args['clients_per_round'] = 10
    args['bs'] = 64
    args['lr'] = 0.1
    args['lr_sched'] = 2
    args['seed'] = 0
    args['input_shape'] = 784
    args['num_class'] = 10
    args['quantize'] = True
    args['quan_level'] = 10
    args['gpu'] = True
    args['gpu'] = args['gpu'] and torch.cuda.is_available()


    args['secure'] = True
    args['secure_epsilon'] = 1.0
    args['secure_delta'] = 10e-4

    args['clipping'] = 2
    args['secure_clip'] = 2

    args['subsampling'] = True
    args['subsampling_gamma'] = .8


    if args['secure']:
        if not args['clipping']:
            args['clipping'] = 1

    set_seed(args['seed'], args['gpu'])

    train_path = os.path.join('./data/mnist/data/train', args['dataset'])
    test_path = os.path.join('./data/mnist/data/test', args['dataset'])

    dataset = read_data(train_path, test_path)

    # for E in [80, 90]:
    #   args['secure'] = False
    #   args['secure_epsilon'] = 0.0
    #   args['secure_delta'] = 0.0
    #   args['subsampling'] = False
    #   args['subsampling_gamma'] = 0.0
    #   args['clipping'] = 0
    #   args['secure_clip'] = 0
    #   args['clients_per_round'] = 10
    #   args['verbose2'] = False
    #   args['local_iters'] = E
    #   args['num_round'] = args['num_iters'] // args['local_iters']

    #   print(json.dumps(args, sort_keys=True, indent=4))
    #   set_seed(args['seed'], args['gpu'])
    #   trainer = Trainer(args, dataset)
    #   trainer.train()

    # for E in [1, 10, 100]:
    #   args['secure'] = False
    #   args['secure_epsilon'] = 0.0
    #   args['secure_delta'] = 0.0
    #   args['subsampling'] = False
    #   args['subsampling_gamma'] = 0.0
    #   args['clipping'] = 0
    #   args['secure_clip'] = 0
    #   args['clients_per_round'] = 10
    #   args['verbose2'] = False
    #   args['local_iters'] = E
    #   args['num_round'] = args['num_iters'] // args['local_iters']

    #   print(json.dumps(args, sort_keys=True, indent=4))
    #   set_seed(args['seed'], args['gpu'])
    #   trainer = Trainer(args, dataset)
    #   trainer.train()

    # for s in [1, 10, 100]:
    #   args['secure'] = False
    #   args['secure_epsilon'] = 0.0
    #   args['secure_delta'] = 0.0
    #   args['subsampling'] = False
    #   args['subsampling_gamma'] = 0.0
    #   args['clipping'] = 0
    #   args['secure_clip'] = 0
    #   args['clients_per_round'] = 10
    #   args['verbose2'] = False
    #   args['quan_level'] = s
    #   if s == 100:
    #     args['quantize'] = False

    #   print(json.dumps(args, sort_keys=True, indent=4))
    #   set_seed(args['seed'], args['gpu'])
    #   trainer = Trainer(args, dataset)
    #   trainer.train()

    # for M in [1, 10, 100]:
    #   args['secure'] = False
    #   args['secure_epsilon'] = 0.0
    #   args['secure_delta'] = 0.0
    #   args['subsampling'] = False
    #   args['subsampling_gamma'] = 0.0
    #   args['clipping'] = 0
    #   args['secure_clip'] = 0
    #   args['clients_per_round'] = M
    #   args['verbose2'] = False

    #   print(json.dumps(args, sort_keys=True, indent=4))
    #   set_seed(args['seed'], args['gpu'])
    #   trainer = Trainer(args, dataset)
    #   trainer.train()

    # for clip in [.5, 1, 1.5, 2]:
    #     args['secure_clip'] = clip
    #     for eps in [0]:
    #         args['secure'] = False
    #         args['secure_epsilon'] = eps
    #         for gamma in [.2, .4, .6, .8]:
    #             args['subsampling_gamma'] = gamma

    #             print(json.dumps(args, sort_keys=True, indent=4))
    #             set_seed(args['seed'], args['gpu'])
    #             trainer = Trainer(args, dataset)
    #             trainer.train()

    # for clip in [.5]:
    #     args['secure_clip'] = clip
    #     for eps in [.5, 1.0, 1.5, 2.0]:
    #         args['secure_epsilon'] = eps
    #         for gamma in [.2, .4, .6, .8]:
    #             args['subsampling_gamma'] = gamma

    #             print(json.dumps(args, sort_keys=True, indent=4))
    #             set_seed(args['seed'], args['gpu'])
    #             trainer = Trainer(args, dataset)
    #             trainer.train()

    for clip in [2.0]:
        args['secure_clip'] = clip
        for eps in [.2]:
            args['secure_epsilon'] = eps
            for gamma in [.6]:
                args['subsampling_gamma'] = gamma

                print(json.dumps(args, sort_keys=True, indent=4))
                set_seed(args['seed'], args['gpu'])
                trainer = Trainer(args, dataset)
                trainer.train()

if __name__ == '__main__':
    main()
