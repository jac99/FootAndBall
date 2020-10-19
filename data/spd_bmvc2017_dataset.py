"""
This module contains functions to process SoccerPlayerDetection_bmvc17_v1 dataset
"Light cascaded convolutional neural networks for accurate player detection"
https://www.cs.ubc.ca/~jhchen14/ccnn_player_detection/
"""

import scipy.io
from PIL import Image
import numpy as np
import os

import torch

import data.augmentation as augmentation
from data.augmentation import PLAYER_LABEL


class SpdDataset(torch.utils.data.Dataset):
    # Read images from the spd_bmvc17_dataset
    def __init__(self, root, ndx, transform):
        """
        Args:
            root: Dataset root
            ndx: Dataset index (1 or 2)
            transform: Optional transform to be applied on a sample
        """

        self.root = root
        self.ndx = ndx
        self.transform = transform
        self.data_path, gt = open_dataset(root, ndx)
        self.image_extension = '.jpg'
        self.image_list = []
        self.gt = []

        gt = gt['annot'][0]
        for bboxes, e in gt:
            image_name = e[0]
            # Verify if the image file exists, if not, skip
            image_path = os.path.join(self.data_path, image_name)
            if not os.path.exists(image_path):
                continue
            self.gt.append(bboxes)
            self.image_list.append(image_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, ndx):
        # Returns transferred image as a normalized tensor
        image_path = self.image_list[ndx]
        image = Image.open(image_path)
        boxes, labels = self.get_annotations(ndx)
        if self.transform is not None:
            image, boxes, labels = self.transform((image, boxes, labels))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels

    def get_annotations(self, ndx):
        # Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        # and torch int64 tensor of corresponding labels

        boxes = []
        labels = []

        # Add annotations for the player position
        for (x1, y1, x2, y2) in self.gt[ndx]:
            boxes.append((x1, y1, x2, y2))
            labels.append(PLAYER_LABEL)

        return np.array(boxes, dtype=np.float), np.array(labels, dtype=np.int64)


def create_spd_dataset(dataset_path, ids, mode):
    # Merge multiple Datasets and then splits them into one training and one validation dataset
    assert mode == 'train' or mode == 'val'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    image_size = (720, 1280)
    if mode == 'train':
        transform = augmentation.TrainAugmentation(size=image_size)
    elif mode == 'val':
        transform = augmentation.NoAugmentation(size=image_size)

    ds_l = []
    for ndx in ids:
        ds_l.append(SpdDataset(dataset_path, ndx, transform))

    dataset = torch.utils.data.ConcatDataset(ds_l)
    return dataset


def open_dataset(root, ndx):
    # Get path to images and ground truth
    print('Reading soccer player detection dataset from: {} set: {}'.format(root, ndx))
    assert os.path.exists(root), print('Dataset root not found: {}'.format(root))
    assert ndx in [1, 2], print('Dataset index can be only 1 or 2')

    gt_path = os.path.join(root, 'annotation_{}.mat'.format(ndx))
    data_path = os.path.join(root, 'DataSet_00{}'.format(ndx))
    assert os.path.exists(gt_path), print('Ground truth not found: {}'.format(gt_path))
    assert os.path.exists(data_path), print('Dataset folder not found: {}'.format(data_path))

    gt = scipy.io.loadmat(gt_path)
    return data_path, gt
