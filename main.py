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
    args['dataset'] = os.getenv('dataset', '4_digits_per_client').replace("'","")
    args['model'] = os.getenv('model', 'logistic').replace("'","")
    #args['model'] = 'logistic'
    #args['model'] = 'cnn'
    #args['model'] = 'mlp'
    #args['model'] = 'snn'
    args['nb_steps'] = int(os.getenv('nb_steps', '10').replace("'",""))
    args['wd'] = 0.001
    args['verbose'] = False
    args['verbose2'] = True
    args['num_iters'] = int(os.getenv('num_iters', '1000').replace("'",""))
    args['local_iters'] = int(os.getenv('local_iters', '10').replace("'",""))
    args['num_round'] = args['num_iters'] // args['local_iters']
    args['clients_per_round'] = int(os.getenv('clients_per_round', '10').replace("'",""))
    args['bs'] = 64
    args['lr'] = float(os.getenv('lr', '0.2').replace("'",""))
    args['lr_sched'] = int(os.getenv('lr_sched', '2').replace("'",""))
    args['seed'] = 0
    args['input_shape'] = 784
    args['num_class'] = 10
    args['quantize'] = True
    args['quan_level'] = int(os.getenv('quant', '10').replace("'",""))
    args['gpu'] = True
    args['gpu'] = args['gpu'] and torch.cuda.is_available()


    args['secure'] = bool(int(os.getenv('secure', '1').replace("'","")))
    args['secure_epsilon'] = float(os.getenv('epsilon', '1.0').replace("'",""))
    args['secure_delta'] = 10e-3

    args['clipping'] = 2
    args['secure_clip'] = float(os.getenv('clip', '2.0').replace("'",""))

    args['subsampling'] = bool(int(os.getenv('subsampling', '1').replace("'","")))
    args['subsampling_gamma'] = float(os.getenv('gamma', '0.8').replace("'",""))


    if args['secure']:
        if not args['clipping']:
            args['clipping'] = 1
    else:
        args['clipping'] = 0
        args['subsampling'] = False

    set_seed(args['seed'], args['gpu'])

    train_path = os.path.join('./data/mnist/data/train', args['dataset'])
    test_path = os.path.join('./data/mnist/data/test', args['dataset'])

    dataset = read_data(train_path, test_path, args)


    print(json.dumps(args, sort_keys=True, indent=4))
    set_seed(args['seed'], args['gpu'])
    trainer = Trainer(args, dataset)
    trainer.train()

if __name__ == '__main__':
    main()
