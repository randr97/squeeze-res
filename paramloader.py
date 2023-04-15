import logging

import torch
import torch.nn as nn
import torch.optim as optim

from models.bottleneck_mish_resnet import ResNetBottleNeckMISH
from models.bottleneck_relu_resnet import ResNetBottleNeckRELU
from models.classic_resnet import ClassicResNet18
from models.lightnet import LightNet

log = logging.getLogger()


class HyperParam:

    def __init__(
        self,
        device,
        pre_epoch,
        epoch,
        lr,
        weight_decay,
        t_max,
        optimizer,
        model='classic_resnet',
        img_channel=3,
        num_classes=10,
        scheduler=True,
        momentum=None,
        *args,
        **kwargs,
    ) -> None:
        # hyper-params-init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = epoch
        self.pre_epoch = pre_epoch

        # model selection
        if model == 'classic_resnet':
            self.model = ClassicResNet18().to(self.device)
        elif model == 'bottleneck_relu_resnet':
            self.model = ResNetBottleNeckRELU().to(self.device)
        elif model == 'bottleneck_mish_resnet':
            self.model = ResNetBottleNeckMISH().to(self.device)
        elif model == 'lightnet':
            self.model = LightNet().to(self.device)
        else:
            raise Exception("Invalid model")

        # optimizer selection
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise Exception("Invalid optimizer")

        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = None
        if scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

    def get_param_count(self):
        ct = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(f"Model Param Count: {ct}")
        return ct
