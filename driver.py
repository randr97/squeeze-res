import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import GenerateDataLoader
from model import ResNet18
from train import Train


class Driver:

    def __init__(
        self,
        device,
        pre_epoch,
        epoch,
        train_batch_size,
        validation_batch_size,
        LR, momentum, weight_decay, T_max, img_channel, num_classes
    ) -> None:
        # hyper-params-init
        self.device = device
        self.epoch = epoch
        self.pre_epoch = pre_epoch
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.model = ResNet18(img_channel, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)

    def run(self):

        Train(
            device=self.device,
            pre_epoch=self.pre_epoch,
            epoch=self.epoch, model=self.model,
            optimizer=self.optimizer, scheduler=self.scheduler,
            criterion=self.criterion,
            train_dataloader=GenerateDataLoader(self.train_batch_size).dataloader(),
            validation_dataloader=GenerateDataLoader(self.validation_batch_size, train=False).dataloader(),
        ).train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # hyper pramas
    parser.add_argument('--pre_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train_batch_size', type=str, default=64)
    parser.add_argument('--validation_batch_size', type=str, default=64)
    parser.add_argument('--LR', type=str, default=0.01)
    parser.add_argument('--momentum', type=str, default=0.9)
    parser.add_argument('--weight_decay', type=str, default=5e-4)
    parser.add_argument('--T_max', type=str, default=200)
    parser.add_argument('--img_channel', type=str, default=3)
    parser.add_argument('--num_classes', type=str, default=10)
    config = parser.parse_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Driver(**vars(config)).run()
