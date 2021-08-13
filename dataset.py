# Reworked class from pytorch (see https://pytorch.org/vision/0.8/_modules/torchvision/datasets/voc.html#VOCDetection)
import os
import torch
import torchvision
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import collections


class ImageFolderWithPaths(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.paths = [f'{root}/{path}' for path in os.listdir(root) if ".jpg" in path]
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        sample = dict()
        name = self.paths[idx][:-4]  # no extension
        sample['idx'] = name

        img = Image.open(self.images[idx]).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)
        sample['img'] = img

        return sample


class LADDDataSET(torchvision.datasets.VisionDataset):
    def __init__(
            self,
            root: str,
            image_set: str,
            transforms: Optional[Callable] = None):     
        super(LADDDataSET, self).__init__(root, transforms=transforms)
        self.image_set = image_set

        voc_root = root
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        description = LADDDataSET.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        # get bounding box coordinates 
        num_objs = len(description['annotation']['object'])
        boxes = []
        for l in description['annotation']['object']:
            bb = l['bndbox']
            boxes.append([int(bb['xmin']), int(bb['ymin']), int(bb['xmax']), int(bb['ymax'])])

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)         # there is only one class            
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def parse_voc_xml(node: ET.Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(LADDDataSET.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
