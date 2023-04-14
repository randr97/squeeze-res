import logging

import torch
import torch.nn as nn

log = logging.getLogger()


class Train:

    def __init__(
        self,
        device,
        pre_epoch,
        epoch,
        model,
        optimizer,
        criterion,
        train_dataloader,
        validation_dataloader,
        scheduler=None,
    ) -> None:
        self.device = device
        self.pre_epoch = pre_epoch
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.training_loss = []
        self.testing_loss = []
        self.training_acc = []
        self.testing_acc = []

    def validate(self):
        with torch.no_grad():
            correct, total, running_loss = 0, 0, 0
            self.model.eval()
            for i, data in enumerate(self.validation_dataloader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            self.testing_loss.append(running_loss / (i + 1))
            self.testing_acc.append(100 * correct / total)
            if self.testing_loss[-1] <= min(self.testing_loss):
                torch.save(self.model.state_dict(), 'model.pth')

    def train(self):
        sum_loss, correct, total = 0, 0, 0
        for eachepoch in range(self.pre_epoch, self.epoch):
            self.model.train()
            for i, data in enumerate(self.train_dataloader):
                # prepare dataset
                # length = len(self.train_dataloader)
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                # forward & backward
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                # ********************* hyper param
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # print ac & loss in each batch
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
            self.training_loss.append(sum_loss / (i + 1))
            self.training_acc.append(100. * correct / total)
            self.validate()
            if self.scheduler:
                self.scheduler.step()
            log.info("*" * 50)
            log.info(
                f"Epoch : {eachepoch + 1}, "
                f"Train Loss: {self.training_loss[-1]:.4f}, Train Acc: {self.training_acc[-1]:.4f}, "
                f"Test Loss: {self.testing_loss[-1]:.4f},  Test Acc: {self.testing_acc[-1]:.4f}"
            )
            log.info("*" * 50)
