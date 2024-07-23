import os
import tarfile

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split


class P100Dataset(Dataset):
    def __init__(self, X, Y, group=None):
        # read in the transforms
        self.if_show_group = True if group is not None else False
        # reshape into 48x48x1
        self.data = X
        self.labels = Y
        self.groups = Y

    # override the length function
    def __len__(self):
        return len(self.data)

    # override the getitem function
    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index].astype(np.float32)
        # load the labels into a list and convert to tensors
        labels = self.labels[index].astype(np.longlong)
        # return data labels
        if self.if_show_group:
            groups = self.groups[index]
            return data, labels, groups
        return data, labels


def prepare_purchase100_dataset(data_dir, k, options=None):
    if options is None:
        options = {'group': None, 'split_train': 0.0}
    split_train = options['split_train']
    # transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize(
    #        mean=[0.4914, 0.4822, 0.4465],
    #        std=[0.2023, 0.1994, 0.2010]
    #    ),
    # ])
    DATASET_PATH = data_dir
    DATASET_NAME = 'dataset_purchase.tgz'
    DATASET_NUMPY = 'data.npz'
    DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

    # Purchase100 Train
    ## Load data
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']

    # Target Dataset
    target_x, shadow_x, target_y, shadow_y = train_test_split(
        X, Y, test_size=0.5, random_state=0
    )
    target_train_x, target_test_x, target_train_y, target_test_y = train_test_split(
        target_x, target_y, test_size=0.5, random_state=0
    )
    # Target Ref
    target_ref = None
    if split_train > 0:
        target_train_x, target_ref_x, target_train_y, target_ref_y = train_test_split(
            target_train_x, target_train_y, test_size=split_train, random_state=0
        )
        target_ref = P100Dataset(
            target_ref_x, target_ref_y,
            group=options['group']
        )
    # target dataset class
    target_train = P100Dataset(
        target_train_x, target_train_y,
        group=options['group']
    )
    target_test = P100Dataset(
        target_test_x, target_test_y,
        group=options['group']
    )
    # Shadow Dataset
    shadow_train_list = []
    shadow_test_list = []
    shadow_ref_list = []
    for i in range(k):
        shadow_train_x, shadow_test_x, shadow_train_y, shadow_test_y = train_test_split(
            shadow_x, shadow_y, test_size=0.5, random_state=i
        )
        if split_train > 0:
            shadow_train_x, shadow_ref_x, shadow_train_y, shadow_ref_y = train_test_split(
                shadow_train_x, shadow_train_y, test_size=split_train, random_state=i
            )
            shadow_ref_list += [
                P100Dataset(
                    shadow_ref_x, shadow_ref_y,
                    group=options['group']
                )
            ]
        shadow_train_list += [
            P100Dataset(
                shadow_train_x, shadow_train_y,
                group=options['group']
            )
        ]
        shadow_test_list += [
            P100Dataset(
                shadow_test_x, shadow_test_y,
                group=options['group']
            )
        ]

    return target_train, target_test, target_ref, shadow_train_list, shadow_test_list, shadow_ref_list


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
