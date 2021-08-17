import yaml
import pandas as pd
import torch
import torchvision
import wandb
import numpy as np
import os
import copy
from tqdm import tqdm
import random
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

from dataset import LADDDataSET, ImageFolderWithPaths
from augmentations import get_transform
from metrics import evaluate_res


def collate_fn(batch):
    """
    This function helps when we have different number of object instances
    in the batches in the dataset.
    """
    return tuple(zip(*batch))


class Runner:
    def __init__(self, params):
        self.params = params
        # train_augs = []
        # if params['transforms']['train']:
        # train_augs.append(A.HorizontalFlip(p=0.5))
        # train_augs.append(A.RandomBrightnessContrast(p=0.3))
        # train_augs.append(A.ShiftScaleRotate(p=0.5))
        # train_augs.append(A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3))
        # A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2)
        # A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)
        # train_augs.append(A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.3))
        # train_augs.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.3))

        # train_augs.append(ToTensorV2())

        # val_augs = []
        # val_augs.append(ToTensorV2())

        # train_transform = A.Compose(train_augs,
        # bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # val_transform = A.Compose(val_augs,
        # bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        train_transform = get_transform(train=True, target_size=(params['transforms']['resize']['h'],
                                                                 params['transforms']['resize']['w']))
        val_transform = get_transform(train=False, target_size=(params['transforms']['resize']['h'],
                                                                params['transforms']['resize']['w']))
        dataset_train = LADDDataSET(params['data_root'],
                                    params['split']['train'],
                                    train_transform)

        dataset_val = LADDDataSET(params['data_root'],
                                  params['split']['val'],
                                  val_transform)

        dataset_test = ImageFolderWithPaths(params['test_root'],
                                            val_transform)

        self.data_loaders = {'train': torch.utils.data.DataLoader(dataset_train,
                                                                  batch_size=params['batch_size'],
                                                                  shuffle=True,
                                                                  num_workers=params['batch_size'],
                                                                  collate_fn=collate_fn),

                             'val': torch.utils.data.DataLoader(dataset_val,
                                                                batch_size=params['batch_size'],
                                                                shuffle=False,
                                                                num_workers=params['batch_size'],
                                                                collate_fn=collate_fn),

                             'test': torch.utils.data.DataLoader(dataset_test,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 num_workers=1,
                                                                 collate_fn=collate_fn)}

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=params['model']['pretrained'],
                                                                          num_classes=2,
                                                                          pretrained_backbone=params['model'][
                                                                              'pretrained_backbone'],
                                                                          min_size=params['model']['min_size'],
                                                                          max_size=params['model']['max_size'],
                                                                          trainable_backbone_layers=params['model'][
                                                                              'trainable_backbone_layers'])

        self.device = torch.device(params['device'])

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=params['optimizer']['lr'],
                                         momentum=params['optimizer']['momentum'],
                                         weight_decay=params['optimizer']['weight_decay'])

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=params['lr_scheduler']['step_size'],
                                                            gamma=params['lr_scheduler']['gamma'])

        self.checkpoints_dir = params['checkpoints_dir']
        self.submissions_dir = params['submissions_dir']

    def train(self):
        self.model.train()

        epoch_metrics = dict()
        epoch_metrics['loss'] = 0.0

        for images, targets in tqdm(self.data_loaders['train']):
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
        cpu_device = torch.device("cpu")
        inference_res = []
        self.model.eval()
        with torch.no_grad():
            for images, targets in tqdm(self.data_loaders['val']):
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                res = targets, outputs
                inference_res.append(res)

        ap_iou0_5, f1_iou0_5 = evaluate_res(inference_res, iou_threshold=0.5,
                                            score_threshold=self.params['score_threshold'])
        ap_iou0_6, f1_iou0_6 = evaluate_res(inference_res, iou_threshold=0.6,
                                            score_threshold=self.params['score_threshold'])

        epoch_metrics = dict()
        epoch_metrics['ap_iou0.5'] = ap_iou0_5
        epoch_metrics['f1_iou0.5'] = f1_iou0_5
        epoch_metrics['ap_iou0.6'] = ap_iou0_6
        epoch_metrics['f1_iou0.6'] = f1_iou0_6

        return epoch_metrics

    def predict(self):
        PATH = f"{self.checkpoints_dir}/{self.params['model_filename']}.pth"
        self.model.load_state_dict(torch.load(PATH))
        self.model.to(self.device)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, sample in tqdm(self.data_loaders['test']):
                idx = sample[0]['name']
                old_width = sample[0]['old_w']
                old_height = sample[0]['old_h']
                new_width = sample[0]['new_w']
                new_height = sample[0]['new_h']

                images = list(image.to(self.device) for image in images)

                predictions = self.model(images)
                boxes = predictions[0]['boxes'].cpu().detach()
                scores = predictions[0]['scores'].cpu().detach()

                scores_ = scores[scores >= self.params['score_threshold']]
                if len(scores_) > 0:
                    threshold = self.params['score_threshold']
                else:
                    threshold = 0.0

                if len(boxes) > 0:
                    # resize reverse
                    boxes[:, [0, 2]] = boxes[:, [0, 2]] * old_width / new_width
                    boxes[:, [1, 3]] = boxes[:, [1, 3]] * old_height / new_height
                    for j, box in enumerate(boxes):
                        xmin = int(box[0].item())
                        ymin = int(box[1].item())
                        xmax = int(box[2].item())
                        ymax = int(box[3].item())
                        score = float(scores[j].item())
                        if score >= threshold:
                            results.append([idx, xmin, ymin, xmax, ymax, score])
                else:
                    results.append([idx, 0, 0, 0, 0, 0.0])

        df = pd.DataFrame(results, columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
        df.to_csv(f"{self.submissions_dir}/{self.params['submission_filename']}.csv", index=False)

    def run(self):
        random.seed(42)
        np.random.seed(42)

        wandb.init(project=self.params['project_name'], config=self.params)

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.submissions_dir, exist_ok=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_ap_iou0_5 = 0

        self.model = self.model.to(self.device)
        for epoch in range(params['num_epochs']):
            train_metrics = self.train()
            self.lr_scheduler.step()
            val_metrics = self.eval()

            logs = {'train': train_metrics,
                    'val': val_metrics,
                    'lr': self.optimizer.param_groups[0]["lr"]}

            wandb.log(logs, step=epoch)

            val_ap_iou0_5 = val_metrics['ap_iou0.5']
            if val_ap_iou0_5 > best_ap_iou0_5:
                best_ap_iou0_5 = val_ap_iou0_5
                best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/{self.params['model_filename']}.pth")
        self.predict()


if __name__ == '__main__':
    with open('./configs/default.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    runner = Runner(params)
    # runner.run()
    runner.predict()

