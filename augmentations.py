import numpy as np
import random
import types
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as Ft


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = Ft.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None, val=False):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
        self.val = val

    def __call__(self, image, target):
        if self.val:
            size = self.min_size
        else:
            size = random.randint(self.min_size, self.max_size)
        image = Ft.resize(image, size)

        target = Ft.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target
    
    def __repr__(self):
        return "RandomResize"
    
class Resize(object):
    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, image, target):

        image = Ft.resize(image, (self.h, self.w))
        target = Ft.resize(target, (self.h, self.w), interpolation=T.InterpolationMode.NEAREST)
        
        return image, target
    
    def __repr__(self):
        return "Resize"


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = Ft.hflip(image)
            target = Ft.hflip(target)
        return image, target
    
    def __repr__(self):
        return "RandomHorizontalFilp"


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = Ft.crop(image, *crop_params)
        target = Ft.crop(target, *crop_params)
        return image, target
    
    def __repr__(self):
        return "RandomCrop"


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = Ft.center_crop(image, self.size)
        target = Ft.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = Ft.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target
    
    def __repr__(self):
        return "ToTensor"


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = Ft.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
    def __repr__(self):
        return "Normalize"
    
class RandomRotate(object):
    def __init__(self, degree=10, prob=0.5):
        self.degree = degree
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            degree = random.randint(-self.degree, self.degree)
            image = Ft.rotate(image, angle=degree)
            target = Ft.rotate(target, angle=degree)
        return image, target
    
    def __repr__(self):
        return "RandomRotate"

class Lambda(object):
    """
    Applies a lambda as a transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, lbl):

        if random.randint(2):

            img = np.array(img, dtype='float')
            img[:, :, 1] *= random.uniform(self.lower, self.upper)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

        return img, lbl
    
class RandomMirror(object):
    def __call__(self, image, label):
        if random.randint(2):
            
            image = np.array(image, dtype='float')
            label = np.array(label, dtype='float')
            image = image[:,::-1]
            label = label[...,None]
            label = label[:,::-1]
            label = label.squeeze()
            image = Image.fromarray(image.astype(np.uint8)).convert('RGB')
            label = Image.fromarray(label.astype(np.uint8))
            
        return image, label


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img, lbl):

        if random.randint(2):
            img = np.array(img, dtype='float')
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
        return img, lbl


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, lbl):

        if random.randint(2):
            img = np.array(img, dtype='float')
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            img = shuffle(img)
            img = Image.fromarray(img).convert('RGB')
        return img, lbl

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."


    def __call__(self, img, lbl):

        if random.randint(2):
            img = np.array(img, dtype='float')
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')

        return img, lbl


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, lbl):

        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, lbl


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels

class SwapChannels(object):


    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image, lbl):

        image = image[:, :, self.swaps]
        return image, lbl

class SegmentationPresetTrain:

    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)  # 520*0.5=260
        max_size = int(2.0 * base_size)  # 520*2=1024

        trans = []
        
        trans.extend([
            RandomResize(min_size, max_size),
            #Resize(520, 520),
            RandomCrop(crop_size),  
            RandomHorizontalFlip(hflip_prob),
            RandomRotate(),
            
            #RandomContrast(),
            #RandomSaturation(),
            #RandomHue(),
            #RandomMirror(),
            
            ToTensor(),
            Normalize(mean=mean, std=std), 
        ])
        self.trans = trans
        self.transforms = Compose(trans) 

    def __call__(self, img, target):
        return self.transforms(img, target)  

class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [
            Resize(base_size, base_size),
            ToTensor(),
            Normalize(mean=mean, std=std), 
        ]
        self.trans = trans
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(crop_size)