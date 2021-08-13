import yaml
import pandas as pd
import torch
import torchvision
import wandb
import numpy as np
import os
import copy
from tqdm import tqdm
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
        voc_root = params['data_root']
        test_root = params['test_root']

        target_size = (params['target_h'], params['target_w'])

        dataset_train = LADDDataSET(voc_root,
                                    params['train_mode'],
                                    get_transform(train=True, target_size=target_size))
        dataset_val = LADDDataSET(voc_root,
                                  params['val_mode'],
                                  get_transform(train=False, target_size=target_size))

        dataset_test = ImageFolderWithPaths(test_root, get_transform(train=False, target_size=target_size))

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

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                          num_classes=2,
                                                                          pretrained_backbone=True,
                                                                          min_size=target_size[0],
                                                                          max_size=target_size[1],
                                                                          trainable_backbone_layers=0)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=params['lr'],
                                         momentum=params['momentum'],
                                         weight_decay=params['weight_decay'])

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=params['step_size'],
                                                            gamma=params['gamma'])

        self.num_epochs = params['num_epochs']
        self.checkpoints_dir = params['checkpoints_dir']

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
        self.model.eval()
        epoch_metrics = dict()
        inference_res = []
        cpu_device = torch.device("cpu")

        with torch.no_grad():
            for images, targets in tqdm(self.data_loaders['val']):
                images = list(img.to(self.device) for img in images)
                outputs = self.model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                res = targets, outputs
                inference_res.append(res)

        ap_iou0_5, f1_iou0_5 = evaluate_res(inference_res, iou_threshold=0.5, score_threshold=0.05)
        ap_iou0_6, f1_iou0_6 = evaluate_res(inference_res, iou_threshold=0.6, score_threshold=0.05)

        epoch_metrics['ap_iou0.5'] = ap_iou0_5
        epoch_metrics['f1_iou0.5'] = f1_iou0_5
        epoch_metrics['ap_iou0.6'] = ap_iou0_6
        epoch_metrics['f1_iou0.6'] = f1_iou0_6

        log = None

        return epoch_metrics, log

    def predict(self):
        results = []
        for sample in self.data_loaders['test']:
            idx = sample['idx']
            img = sample['img']
            images = [img.to(self.device)]
            predictions = self.model(images)
            boxes = predictions[0]['boxes']
            scores = predictions[0]['scores']
            for j, box in enumerate(boxes):
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                score = scores[j]
                results.append([idx, xmin, ymin, xmax, ymax, score])

        df = pd.DataFrame(results, columns=['id', 'xmin', 'ymin', 'xmax', 'ymax', 'score'])
        df.to_csv(f"{self.params['submissions_dir']}/{self.params['submission_filename']}.csv", index=False)

    def run(self):
        wandb.init(project=self.params['project_name'], config=self.params)
        np.random.seed(0)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_ap_iou0_5 = 1

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            train_metrics = self.train()
            self.lr_scheduler.step()
            val_metrics, log = self.eval()

            logs = {'train': train_metrics,
                    'val': val_metrics}
            wandb.log(logs, step=epoch)

            val_ap_iou0_5 = val_metrics['ap_iou0.5']
            if val_ap_iou0_5 < best_ap_iou0_5:
                best_ap_iou0_5 = val_ap_iou0_5
                best_model_wts = copy.deepcopy(self.model.state_dict())

                # wandb.log({f"epoch = {epoch}": [wandb.Image(log['image'], caption=f"({log['dy']},{log['dx']})")]})

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/resnet50_FRCNN_baseline.pth")
        self.predict()


if __name__ == '__main__':
    with open('./configs/default.yaml', 'r') as file:
        params = yaml.load(file, yaml.Loader)

    runner = Runner(params)
    runner.run()
