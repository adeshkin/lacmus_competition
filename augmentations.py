import random
import torchvision.transforms.functional as F


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            if bbox.shape[0] > 0:
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target


class Resize(object):
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, image, target):
        old_height, old_width = image.shape[-2:]
        image = F.resize(image, self.target_size, interpolation=F.InterpolationMode.BILINEAR)
        new_height, new_width = image.shape[-2:]
        #bbox = target["boxes"]
        #if bbox.shape[0] > 0:
        #    bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_width / old_width
        #    bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_height / old_height
        #    target["boxes"] = bbox

        return image, target


def get_transform(train, target_size):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    # transforms.append(Resize(target_size))
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
