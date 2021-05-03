import torch
import numpy as np
import pickle
import os
import torchvision
cpath = os.path.dirname(__file__)


NUM_USER = 100
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = False
np.random.seed(6)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def choose_digits(split_data_lst, is_last_two, NUM_DIGITS_PER_USER):
    available_digit = []
    num_remaining = np.zeros(10)
    for i, digit in enumerate(split_data_lst):
        num_remaining[i] = len(digit)
        if len(digit) > 0:
            available_digit.append(i)

    # if not is_last_two:
    #     for j in available_digit:
    #         if num_remaining[j] <= 1:
    #             available_digit.remove(j)

    num_pick = np.min((NUM_DIGITS_PER_USER, len(available_digit)))

    try:
        lst = np.random.choice(available_digit, num_pick, replace=False).tolist()

    except:
        print(available_digit)
    return lst


def main():
    # Get MNIST data, normalize, and divide by level
    print('>>> Get MNIST data.')
    trainset = torchvision.datasets.MNIST(DATASET_FILE, download=True, train=True)
    testset = torchvision.datasets.MNIST(DATASET_FILE, download=True, train=False)

    train_mnist = ImageDataset(trainset.train_data, trainset.train_labels)
    test_mnist = ImageDataset(testset.test_data, testset.test_labels)

    for NUM_DIGITS_PER_USER in range(1, 11):

        print("\nCreating dataset HET_MNIST({})".format(NUM_DIGITS_PER_USER))

        NUM_SPLIT = NUM_USER * NUM_DIGITS_PER_USER // 10

        mnist_traindata = []
        for number in range(10):
            idx = train_mnist.target == number
            mnist_traindata.append(train_mnist.data[idx])
        split_mnist_traindata = []
        for digit in mnist_traindata:
            split_mnist_traindata.append(data_split(digit, NUM_SPLIT))

        mnist_testdata = []
        for number in range(10):
            idx = test_mnist.target == number
            mnist_testdata.append(test_mnist.data[idx])
        split_mnist_testdata = []
        for digit in mnist_testdata:
            split_mnist_testdata.append(data_split(digit, NUM_SPLIT))

        data_distribution = np.array([len(v) for v in mnist_traindata])
        data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
        print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

        digit_count = np.array([len(v) for v in split_mnist_traindata])
        print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

        digit_count = np.array([len(v) for v in split_mnist_testdata])
        print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

        # Assign train samples to each user
        train_X = [[] for _ in range(NUM_USER)]
        train_y = [[] for _ in range(NUM_USER)]
        test_X = [[] for _ in range(NUM_USER)]
        test_y = [[] for _ in range(NUM_USER)]

        for user in range(NUM_USER):
            print(user, np.array([len(v) for v in split_mnist_traindata]))

            if user < NUM_USER - 2:
                is_last_two = False
            else:
                is_last_two = True

            for d in choose_digits(split_mnist_traindata, is_last_two, NUM_DIGITS_PER_USER):
                l = len(split_mnist_traindata[d][-1])
                train_X[user] += split_mnist_traindata[d].pop().tolist()
                train_y[user] += (d * np.ones(l)).tolist()

                l = len(split_mnist_testdata[d][-1])
                test_X[user] += split_mnist_testdata[d].pop().tolist()
                test_y[user] += (d * np.ones(l)).tolist()

        print('>>> Set data path for MNIST.')
        train_path = '{}/data/train/{}_digits_per_client/{}_digits_per_client.pkl'.format(cpath, NUM_DIGITS_PER_USER, NUM_DIGITS_PER_USER)
        test_path = '{}/data/test/{}_digits_per_client/{}_digits_per_client.pkl'.format(cpath, NUM_DIGITS_PER_USER, NUM_DIGITS_PER_USER)

        dir_path = os.path.dirname(train_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        dir_path = os.path.dirname(test_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Create data structure
        train_data = {'users': [], 'user_data': {}, 'num_samples': []}
        test_data = {'users': [], 'user_data': {}, 'num_samples': []}

        for i in range(NUM_USER):
            train_data['users'].append(i)
            train_data['user_data'][i] = {'x': train_X[i], 'y': train_y[i]}
            train_data['num_samples'].append(len(train_X[i]))

            test_data['users'].append(i)
            test_data['user_data'][i] = {'x': test_X[i], 'y': test_y[i]}
            test_data['num_samples'].append(len(test_X[i]))

        print('>>> User data distribution: {}'.format(train_data['num_samples']))
        print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
        print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

        if SAVE:
            with open(train_path, 'wb') as outfile:
                pickle.dump(train_data, outfile)
            with open(test_path, 'wb') as outfile:
                pickle.dump(test_data, outfile)

            print('>>> Save data.')


if __name__ == '__main__':
    main()
