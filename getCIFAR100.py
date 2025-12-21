import os
import pickle
import shutil

from PIL import Image
import numpy as np

import torchvision
from tqdm import tqdm


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


def get_images(cifar100_path):
    torchvision.datasets.CIFAR100(root=cifar100_path, train=True, download=True)
    torchvision.datasets.CIFAR100(root=cifar100_path, train=False, download=True)

    cifar100_path = os.path.join(cifar100_path, "cifar-100-python")
    save_path = os.path.join(cifar100_path, 'for_train')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    meta_dict = unpickle(os.path.join(cifar100_path, 'meta'))
    train_dict = unpickle(os.path.join(cifar100_path, 'train'))
    test_dict = unpickle(os.path.join(cifar100_path, 'test'))

    for data_type in ["train", "test"]:
        if data_type == "train":
            data_dict = train_dict
        elif data_type == "test":
            data_dict = test_dict

        for fine_label_name in meta_dict['fine_label_names']:
            os.makedirs(os.path.join(save_path, data_type, fine_label_name), exist_ok=True)

        for i in tqdm(range(data_dict['data'].shape[0])):
            img = data_dict['data'][i]
            img = np.reshape(img, (3, 32, 32))
            img = img.transpose((1, 2, 0))
            img = Image.fromarray(img)
            fine_label_name = meta_dict['fine_label_names'][data_dict['fine_labels'][i]]
            filename = data_dict['filenames'][i]
            filepath = os.path.join(save_path, data_type, fine_label_name, filename)
            img.save(filepath)


# 设置数据集路径
cifar100_path = "."

get_images(cifar100_path)
