import os
import random
import torch
import numpy as np
from randaugment import RandAugment
import torchvision.transforms as transforms
from PIL import ImageDraw
from dataset.handlers import COCO2014_handler, VOC2012_handler, NUS_WIDE_handler

np.set_printoptions(suppress=True)

HANDLER_DICT = {
    'voc': VOC2012_handler,
    'coco': COCO2014_handler,
    'nus': NUS_WIDE_handler,
}

def get_datasets(args):

    print('Getting datasets...')
    
    nrs = np.random.RandomState(args.seed)

    train_transform = TransformWithPatches_Train(args)
    test_transform = TransformWithPatches_Val(args)
    
    source_data = load_data(args.dataset_dir)

    train_labels = source_data['train']['labels']
    n_train = len(train_labels)
    n_lb = int(args.lb_ratio*n_train)
    indices = nrs.permutation(n_train)
    lb_idxs = indices[:n_lb]
    ub_idxs = indices[n_lb:]

    print(n_train, len(lb_idxs), len(ub_idxs))

    lb_train_imgs, lb_train_labels = source_data['train']['images'][lb_idxs], source_data['train']['labels'][lb_idxs]
    ub_train_imgs, ub_train_labels = source_data['train']['images'][ub_idxs], source_data['train']['labels'][ub_idxs]

    data_handler = HANDLER_DICT[args.dataset_name]
    lb_train_dataset = data_handler(lb_train_imgs, lb_train_labels, args.dataset_dir, transform=train_transform)
    ub_train_dataset = data_handler(ub_train_imgs, ub_train_labels, args.dataset_dir, transform=train_transform)

    val_dataset = data_handler(source_data['val']['images'], source_data['val']['labels'], args.dataset_dir, transform=test_transform)

    
    return lb_train_dataset, ub_train_dataset, val_dataset

def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
    return data


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class TransformWithPatches_Train(object):
    def __init__(self, args):
        self.grid_perside = args.grid_perside
        self.img_size = args.img_size
        
        self.weak = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.Resize((args.img_size, args.img_size)),
			transforms.ToTensor()])
        self.weak_patch = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.Resize((args.img_size, args.img_size)),
			transforms.ToTensor()])

        strong = [transforms.RandomHorizontalFlip(),
                  transforms.Resize((args.img_size, args.img_size)),
                  RandAugment(),
                  transforms.ToTensor()]
        strong_patch = [transforms.RandomHorizontalFlip(),
                  transforms.Resize((args.img_size, args.img_size)),
                  RandAugment(),
                  transforms.ToTensor()]

        if args.cutout>0:
            strong.insert(2, CutoutPIL(cutout_factor=args.cutout))
            strong_patch.insert(2, CutoutPIL(cutout_factor=args.cutout))

        self.strong = transforms.Compose(strong)
        self.strong_patch = transforms.Compose(strong_patch)

    def __call__(self, img):
        weak_list = [self.weak(img)]
        strong_list = [self.strong(img)] 

        # Append patches
        img = img.resize((self.img_size, self.img_size))

        # To permute the order for local patched
        x_order = np.random.permutation(self.grid_perside)
        y_order = np.random.permutation(self.grid_perside)

        grid_size_x = img.size[0] // self.grid_perside
        grid_size_y = img.size[1] // self.grid_perside
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                weak_list.append(self.weak_patch(patch))
                strong_list.append(self.strong_patch(patch))
        
        return weak_list, strong_list


class TransformWithPatches_Val(object):
    def __init__(self, args):
        self.grid_perside = args.grid_perside
        self.img_size = args.img_size
        
        self.weak = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()])
        self.weak_patch = transforms.Compose([
			transforms.Resize((args.img_size, args.img_size)),
			transforms.ToTensor()])

    def __call__(self, img):
        weak_list = [self.weak(img)]

        # Append patches
        img = img.resize((self.img_size, self.img_size))

        # To permute the order for local patched
        x_order = np.random.permutation(self.grid_perside)
        y_order = np.random.permutation(self.grid_perside)

        grid_size_x = img.size[0] // self.grid_perside
        grid_size_y = img.size[1] // self.grid_perside
        
        for i in x_order:
            for j in y_order:
                x_offset = i * grid_size_x
                y_offset = j * grid_size_y
                patch = img.crop((x_offset, y_offset, x_offset + grid_size_x, y_offset + grid_size_y))
                # Append patches
                weak_list.append(self.weak_patch(patch))
        
        return weak_list