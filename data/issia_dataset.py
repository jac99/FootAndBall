# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import torch
from PIL import Image
import numpy as np
import os

import data.augmentation as augmentation
import data.issia_utils as issia_utils
from data.augmentation import BALL_BBOX_SIZE, BALL_LABEL, PLAYER_LABEL

from torch.utils.data import Sampler


class IssiaDataset(torch.utils.data.Dataset):
    # Read images from the ISSIA dataset
    def __init__(self, dataset_path, cameras, transform, only_ball_frames=False):
        """
        Args:
            root_dir: Directory with all the images
            camera_id: Camera id between 1 and 6
            transform: Optional transform to be applied on a sample
        """
        for camera_id in cameras:
            assert 1 <= camera_id <= 6, 'Unknown camera id: {}'.format(camera_id)

        self.dataset_path = dataset_path
        self.cameras = cameras
        self.transform = transform
        self.only_ball_frames = only_ball_frames
        self.image_extension = '.png'
        # Dictionary with ground truth annotations per camera
        self.gt_annotations = {}
        # list of images as tuples (image_path, camera_id, image index)
        self.image_list = []

        unpack_path = os.path.join(dataset_path, 'unpacked')
        if not os.path.exists(unpack_path):
            os.mkdir(unpack_path)

        for camera_id in cameras:
            # Extract frames from the sequence if needed
            frames_path = os.path.join(dataset_path, 'unpacked', str(camera_id))
            self.frames_path = frames_path
            if not os.path.exists(frames_path):
                os.mkdir(frames_path)
                issia_utils.extract_frames(dataset_path, camera_id, frames_path)

            # Read ground truth data for the sequence
            self.gt_annotations[camera_id] = issia_utils.read_issia_ground_truth(camera_id, dataset_path)

            # Create a list with ids of all images with any annotation
            if self.only_ball_frames:
                annotated_frames = set(self.gt_annotations[camera_id].ball_pos)
            else:
                annotated_frames = set(self.gt_annotations[camera_id].ball_pos) and set(
                    self.gt_annotations[camera_id].persons)

            min_annotated_frame = min(annotated_frames)
            # Skip the first 50 annotated frames - as they may contain wrong annotations
            annotated_frames = [e for e in list(annotated_frames) if e > min_annotated_frame+50]

            for e in annotated_frames:
                # Verify if the image file exists
                # Sometimes ground truth contains more annotations than images in the sequence
                file_path = os.path.join(frames_path, str(e) + self.image_extension)
                if os.path.exists(file_path):
                    self.image_list.append((file_path, camera_id, e))

        self.n_images = len(self.image_list)
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = set([ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx])
        print('ISSIA CNR: {} frames with the ball'.format(len(self.ball_images_ndx)))
        print('ISSIA CNR: {} frames without the ball'.format(len(self.no_ball_images_ndx)))

    def __len__(self):
        return self.n_images

    def __getitem__(self, ndx):
        # Returns transferred image as a normalized tensor
        image_path, camera_id, image_ndx = self.image_list[ndx]
        image = Image.open(image_path)
        boxes, labels = self.get_annotations(camera_id, image_ndx)
        image, boxes, labels = self.transform((image, boxes, labels))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return image, boxes, labels

    def get_annotations(self, camera_id, image_ndx):
        # Prepare annotations as list of boxes (xmin, ymin, xmax, ymax) in pixel coordinates
        # and torch int64 tensor of corresponding labels
        boxes = []
        labels = []

        # Add annotations for the ball position: positions of the ball centre
        ball_pos = self.gt_annotations[camera_id].ball_pos[image_ndx]
        for (x, y) in ball_pos:
            x1 = x - BALL_BBOX_SIZE // 2
            x2 = x1 + BALL_BBOX_SIZE
            y1 = y - BALL_BBOX_SIZE // 2
            y2 = y1 + BALL_BBOX_SIZE
            boxes.append((x1, y1, x2, y2))
            labels.append(BALL_LABEL)

        # Add annotations for the player position
        for (player_id, player_height, player_width, player_x, player_y) in self.gt_annotations[camera_id].persons[image_ndx]:
            boxes.append((player_x, player_y, player_x + player_width, player_y + player_height))
            labels.append(PLAYER_LABEL)

        return np.array(boxes, dtype=np.float), np.array(labels, dtype=np.int64)

    def get_elems_with_ball(self):
        # Get indexes of images with ball ground truth
        ball_images_ndx = []
        for ndx, (_, camera_id, image_ndx) in enumerate(self.image_list):
            ball_pos = self.gt_annotations[camera_id].ball_pos[image_ndx]
            if len(ball_pos) > 0:
                ball_images_ndx.append(ndx)

        return ball_images_ndx


def create_issia_dataset(dataset_path, cameras, mode, only_ball_frames=False):
    # Get ISSIA datasets for multiple cameras
    assert mode == 'train' or mode == 'val'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    train_image_size = (720, 1280)
    val_image_size = (1080, 1920)
    if mode == 'train':
        transform = augmentation.TrainAugmentation(size=train_image_size)
    elif mode == 'val':
        transform = augmentation.NoAugmentation(size=val_image_size)

    dataset = IssiaDataset(dataset_path, cameras, transform, only_ball_frames=only_ball_frames)
    return dataset
