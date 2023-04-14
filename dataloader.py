import torch
import torchvision
from torchvision import transforms


class GenerateDataLoader:

    def __init__(self, batch_size, train=True, download=True, num_workers=2):
        self.train = train
        self.batch_size = batch_size
        self.shuffle = train
        self.download = download
        self.num_workers = num_workers
        self.compose_params = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ] if train else [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]

    def dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='../data',
                train=self.train,
                download=self.download,
                transform=transforms.Compose(self.compose_params)
            ),
            batch_size=self.batch_size,
            shuffle=self.train,
            num_workers=self.num_workers)
