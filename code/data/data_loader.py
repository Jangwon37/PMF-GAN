import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

class DataLoad:
    def __init__(self):
        pass

    def load_data_mnist(self, batch_size=128):
        data_path = "../../../data/mnist"
        os.makedirs(data_path, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def load_data_f_mnist(self, batch_size=128):
        data_path = "../../../data/f_mnist"
        os.makedirs(data_path, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def load_data_cifar10(self, batch_size=128):
        data_path = "../../../data/cifar-10-python"
        os.makedirs(data_path, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def load_data_cifar100(self, batch_size=128):
        data_path = "../../../data/cifar-100-python"
        os.makedirs(data_path, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def load_data_celeba(self, batch_size=128, image_size=64):
        data_dir = "../../../data/celeba"
        os.makedirs(data_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image_folder = os.path.join(data_dir, 'img_align_celeba')
        dataset = datasets.ImageFolder(root=image_folder, transform=transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def load_data_lsun(self, batch_size=128, image_size=64):
        data_dir = "../../../data/lsun"
        os.makedirs(data_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image_folder = os.path.join(data_dir, 'data0/lsun/bedroom')
        dataset = datasets.ImageFolder(root=image_folder, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def load_data_afhq(self, batch_size=64, image_size=128):
        data_dir = "../../../data/afhq"
        os.makedirs(data_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image_folder = os.path.join(data_dir, 'train')
        dataset = datasets.ImageFolder(root=image_folder, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
