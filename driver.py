import argparse
import logging
import json

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import GenerateDataLoader
from paramloader import HyperParam
from train import Train

# log settings
logging.basicConfig(filename='logs.log', level=logging.INFO)
log = logging.getLogger()
console_handler = logging.StreamHandler()
log_format = '%(asctime)s | %(levelname)s: %(message)s'
console_handler.setFormatter(logging.Formatter(log_format))
log.addHandler(console_handler)


class Driver:

    def __init__(self, config) -> None:
        self.config = config

    def run(self):

        train_params = HyperParam(**vars(self.config))
        # Generate run details
        run_details = {**vars(train_params)}
        run_details.update({'model_params_count': train_params.get_param_count()})
        run_details.update(**vars(self.config))
        del run_details['criterion']
        log.info(f"Traing Res with params: {run_details}")
        with open('run_details.json', 'w') as f:
            json.dump(run_details, f)
        trainer = Train(
            **vars(train_params),
            train_dataloader=GenerateDataLoader(self.config.train_batch_size).dataloader(),
            validation_dataloader=GenerateDataLoader(self.config.validation_batch_size, train=False).dataloader(),
        )
        trainer.train()
        run_details.update({
            'training_loss': trainer.training_loss,
            'testing_loss': trainer.testing_loss,
            'training_acc': [i.item() for i in trainer.training_acc],
            'testing_acc': [i.item() for i in trainer.testing_acc],
        })
        with open('run_details.json', 'w') as f:
            json.dump(run_details, f)


if __name__ == '__main__':
    log.info('Started')
    parser = argparse.ArgumentParser()
    # hyper pramas
    parser.add_argument('--pre_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--validation_batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--t_max', type=int, default=200)
    parser.add_argument('--img_channel', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--model', type=str, default='classic_resnet')
    parser.add_argument('--scheduler', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cuda")
    config = parser.parse_args()
    Driver(config).run()
    log.info('End')
