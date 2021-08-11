import yaml
import torch
import torchvision
import wandb
import numpy as np
import os
import copy
from dataset import LADDDataSET
from augmentations import get_transform


def collate_fn(batch):
    """
    This function helps when we have different number of object instances
    in the batches in the dataset.
    """
    return tuple(zip(*batch))


class Runner:
    def __init__(self, params):
        voc_root = '/home/cds-k/Desktop/lacmus/data_lacmus/TrainingData'

        dataset_train = LADDDataSET(voc_root,
                                    'train_non_empty',
                                    get_transform(train=True, target_size=params['target_size']))
        dataset_val = LADDDataSET(voc_root,
                                  'val',
                                  get_transform(train=False, target_size=params['target_size']))

        self.data_loaders = {'train': torch.utils.data.DataLoader(dataset_train,
                                                                  batch_size=params['batch_size'],
                                                                  shuffle=True,
                                                                  num_workers=4,
                                                                  collate_fn=collate_fn),

                             'val': torch.utils.data.DataLoader(dataset_val,
                                                                batch_size=params['batch_size'],
                                                                shuffle=False,
                                                                num_workers=4,
                                                                collate_fn=collate_fn)}

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                          num_classes=2,
                                                                          pretrained_backbone=True,
                                                                          min_size=params['target_size'][0],
                                                                          max_size=params['target_size'][1],
                                                                          trainable_backbone_layers=0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=params['lr'],
                                         momentum=0.9,
                                         weight_decay=0.0005)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=3,
                                                            gamma=0.1)

        self.num_epochs = params['num_epochs']

        self.checkpoints_dir = params['checkpoints_dir']

    def train(self):
        self.model.train()

        epoch_metrics = dict()
        epoch_metrics['loss'] = 0.0

        for images, targets in self.data_loaders['train']:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()
            losses.backward()

            self.optimizer.step()

            epoch_metrics['loss'] += losses.cpu().detach()

        for m in epoch_metrics:
            epoch_metrics[m] = epoch_metrics[m] / len(self.data_loaders['train'])

        return epoch_metrics

    def eval(self):
        self.model.eval()
        epoch_metrics = dict()
        inference_res = []
        cpu_device = torch.device("cpu")

        with torch.no_grad():
            for images, targets in self.data_loaders['val']:
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                res = targets, outputs
                inference_res.append(res)

        average_precision, F1 = evaluate_res(inference_res, iou_threshold=0.5, score_threshold=0.05)
        average_precision, F1 = evaluate_res(inference_res, iou_threshold=0.6, score_threshold=0.05)

        for m in epoch_metrics:
            epoch_metrics[m] = epoch_metrics[m] / len(self.data_loaders['val'])

        log = None

        return epoch_metrics, log

    def run(self):
        wandb.init(project=self.params['project_name'], params=self.params)

        np.random.seed(0)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            train_metrics = self.train()
            self.lr_scheduler.step()
            val_metrics, log = self.eval()

            logs = {'train': train_metrics,
                    'val': val_metrics}
            wandb.log(logs, step=epoch)

            val_loss = val_metrics['loss']
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

                # wandb.log({f"epoch = {epoch}": [wandb.Image(log['image'], caption=f"({log['dy']},{log['dx']})")]})

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/resnet50_FRCNN_baseline.pth")


if __name__ == '__main__':
    with open('./configs/default.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    runner = Runner(params)
    runner.run()
