import torchvision
import torch
from torchvision import datasets, transforms


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class mnist_dataset(Dataset):
    def __init__(self, training=True, data_dir="./dataset"):
        super(mnist_dataset, self).__init__()

        self.train = training
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5], std=[0.5])])

        self.train_data = datasets.MNIST(root=data_dir,
                                    transform=transform,
                                    train=True,
                                    download=True)

        self.test_data = datasets.MNIST(root=data_dir,
                                   transform=transform,
                                   train=False)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            input_data = self.train_data.data[index]
            label = self.train_data.targets[index]

            sample = {'input': input_data, 'label': label}
        else:
            input_data = self.test_data.data[index]
            label = self.test_data.targets[index]

            sample = {'input': input_data, 'label': label}

        return sample
