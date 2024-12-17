import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def get_bMNIST(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    # Load MNIST dataset
    trainset = torchvision.datasets.MNIST(root='./data/raw', train=True, download=True,
                                        transform=transform)
    testset = torchvision.datasets.MNIST(root='./data/raw', train=False, download=True,
                                        transform=transform)

    # Tranform into binary MNIST
    # Classes 0, ..., 4 maps to 0
    # Classes 5, ..., 9 maps to 1
    trainset.targets = (trainset.targets >= 5).type(torch.float32)
    testset.targets = (testset.targets >= 5).type(torch.float32)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader