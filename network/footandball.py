# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import torch
import torch.nn as nn

import network.fpn as fpn
import network.nms as nms
from data.augmentation import BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE


# Get ranges of cells to mark with ground truth location
def get_active_cells(bbox_center_x, bbox_center_y, downsampling_factor, conf_width, conf_height, delta):
    cell_x = int(bbox_center_x / downsampling_factor)
    cell_y = int(bbox_center_y / downsampling_factor)
    x1 = max(cell_x - delta // 2, 0)
    x2 = min(cell_x + delta // 2, conf_width - 1)
    y1 = max(cell_y - delta // 2, 0)
    y2 = min(cell_y + delta // 2, conf_height - 1)
    return x1, y1, x2, y2


def cell2pixel(cell_x, cell_y, downsampling_factor):
    # Inverse function to get_active_cells
    # Returns a range of pixels corresponding to the given cell
    x1 = cell_x * downsampling_factor
    x2 = cell_x * downsampling_factor + downsampling_factor - 1
    y1 = cell_y * downsampling_factor
    y2 = cell_y * downsampling_factor + downsampling_factor - 1
    return x1, y1, x2, y2


def create_groundtruth_maps(bboxes, blabels, img_shape, player_downsampling_factor, ball_downsampling_factor,
                            player_delta, ball_delta):
    # Generate ground truth: player location map, player confidence map and ball confidence map
    # targets: List of ground truth player and ball positions
    # img_shape: shape of the input image
    # ball_delta: number of cells marked around the bbox center (must be an odd number: 1, 3, 5, ....)
    # player_delta: number of cells marked around the bbox center (must be an odd number: 1, 3, 5, ....)

    # Number of elements in the minibatch
    num = len(bboxes)

    h, w = img_shape
    # Size of target confidence maps
    ball_conf_height = h // ball_downsampling_factor
    ball_conf_width = w // ball_downsampling_factor
    player_conf_height = h // player_downsampling_factor
    player_conf_width = w // player_downsampling_factor

    # match priors (default boxes) and ground truth boxes
    player_loc_t = torch.zeros([num, player_conf_height, player_conf_width, 4], dtype=torch.float)
    player_conf_t = torch.zeros([num, player_conf_height, player_conf_width], dtype=torch.long)
    ball_conf_t = torch.zeros([num, ball_conf_height, ball_conf_width], dtype=torch.long)

    for idx, (boxes, labels) in enumerate(zip(bboxes, blabels)):
        # Iterate over all batch elements
        for box, label in zip(boxes, labels):
            # Iterate over all objects in a single frame
            bbox_center_x = (box[0] + box[2]) / 2.
            bbox_center_y = (box[1] + box[3]) / 2.
            bbox_width = box[2] - box[0]
            bbox_height = box[3] - box[1]

            if label == BALL_LABEL:
                # Convert bbox centers to cell coordinates in the ball confidence map
                x1, y1, x2, y2 = get_active_cells(bbox_center_x, bbox_center_y, ball_downsampling_factor,
                                                  ball_conf_width, ball_conf_height, ball_delta)
                ball_conf_t[idx, y1:y2 + 1, x1:x2 + 1] = 1
            elif label == PLAYER_LABEL:
                # Convert bbox centers to cell coordinates in the player confidence map
                x1, y1, x2, y2 = get_active_cells(bbox_center_x, bbox_center_y, player_downsampling_factor,
                                                  player_conf_width, player_conf_height, player_delta)
                player_conf_t[idx, y1:y2 + 1, x1:x2 + 1] = 1

                # Ground truth for the player bounding box
                # We encode bounding box as relative position of the centre (with respect to the cell centre)
                # and it's width and height in normalized coordinates in [0..1] range (where 1 is image width/height)

                # pixel coordinates of each cell center
                temp_x = torch.tensor(range(x1, x2 + 1)).float() * player_downsampling_factor + \
                         (player_downsampling_factor - 1) / 2
                temp_y = torch.tensor(range(y1, y2 + 1)).float() * player_downsampling_factor + \
                         (player_downsampling_factor - 1) / 2

                # Displacement of the bbox center from the cell center in relative coordinates
                temp_x = (bbox_center_x - temp_x) / w
                temp_y = (bbox_center_y - temp_y) / h

                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 0] = temp_x.unsqueeze(0)
                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 1] = temp_y.unsqueeze(1)

                # Normalized width and height
                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 2] = bbox_width / w
                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 3] = bbox_height / h

    return player_loc_t, player_conf_t, ball_conf_t


def count_parameters(model):
    # Count number of parameters in the network: all and trainable
    # Return tuple (all_parametes, trainable_parameters)
    if model is None:
        return 0, 0
    else:
        ap = sum(p.numel() for p in model.parameters())
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return ap, tp


class FootAndBall(nn.Module):
    def __init__(self, phase, base_network: nn.Module, player_regressor: nn.Module, player_classifier: nn.Module,
                 ball_classifier: nn.Module, max_player_detections=100, max_ball_detections=100, player_threshold=0.0,
                 ball_threshold=0.0):
        # phase: in 'train' returns unnormalized confidence values in feature maps (logits)
        #        in 'eval' returns normalized confidence values (passed through Softmax)
        #        in 'detect' returns detection bounding boxes
        # max_player_detections, max_ball_detections, player_threshold, ball_threshold: used
        #        only in 'detect' mode
        super(FootAndBall, self).__init__()

        assert phase in ['train', 'eval', 'detect']

        self.phase = phase
        self.base_network = base_network
        self.ball_classifier = ball_classifier
        self.player_classifier = player_classifier
        self.player_regressor = player_regressor
        self.max_player_detections = max_player_detections
        self.max_ball_detections = max_ball_detections
        self.player_threshold = player_threshold
        self.ball_threshold = ball_threshold

        # Downsampling factor for ball and player feature maps
        self.ball_downsampling_factor = 4
        self.player_downsampling_factor = 16
        # Number of cells marked around the bbox center
        # By default we mark 1x1 cells for players (each cell having 16x16 pixels) and 3x3 cells for ball (each cell
        # having 4x4 pixels)
        self.ball_delta = 3
        self.player_delta = 3

        self.softmax = nn.Softmax(dim=1)
        self.nms_kernel_size = (3, 3)
        self.nms = nms.NonMaximaSuppression2d(self.nms_kernel_size)

    def detect_from_map(self, confidence_map, downscale_factor, max_detections, bbox_map=None):
        # downscale_factor: downscaling factor of the confidence map versus an original image

        # Confidence map is [B, C=2, H, W] tensor, where C=0 is background and C=1 is an object
        confidence_map = self.nms(confidence_map)[:, 1]
        # confidence_map is (B, H, W) tensor
        batch_size, h, w = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
        confidence_map = confidence_map.view(batch_size, -1)

        values, indices = torch.sort(confidence_map, dim=-1, descending=True)
        if max_detections < indices.shape[1]:
            indices = indices[:, :max_detections]

        # Compute indexes of cells with detected object and convert to pixel coordinates
        xc = indices % w
        xc = xc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        yc = torch.div(indices, w, rounding_mode='trunc')
        yc = yc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        # Bounding boxes are encoded as a relative position of the centre (with respect to the cell centre)
        # and it's width and height in normalized coordinates (where 1 is the width/height of the player
        # feature map)
        # Position x and y of the bbox centre offset in normalized coords
        # (dx, dy, w, h)

        if bbox_map is not None:
            # bbox_map is (B, C=4, H, W) tensor
            bbox_map = bbox_map.view(batch_size, 4, -1)
            # bbox_map is (B, C=4, H*W) tensor
            # Convert from relative to absolute (in pixel) values
            bbox_map[:, 0] *= w * downscale_factor
            bbox_map[:, 2] *= w * downscale_factor
            bbox_map[:, 1] *= h * downscale_factor
            bbox_map[:, 3] *= h * downscale_factor
        else:
            # For the ball bbox map is not given. Create fixed-size bboxes
            batch_size, h, w = confidence_map.shape[0], confidence_map.shape[-2], confidence_map.shape[-1]
            bbox_map = torch.zeros((batch_size, 4, h * w), dtype=torch.float).to(confidence_map.device)
            bbox_map[:, [2, 3]] = BALL_BBOX_SIZE

        # Resultant detections (batch_size, max_detections, bbox),
        # where bbox = (x1, y1, x2, y2, confidence) in pixel coordinates
        detections = torch.zeros((batch_size, max_detections, 5), dtype=float).to(confidence_map.device)

        for n in range(batch_size):
            temp = bbox_map[n, :, indices[n]]
            # temp is (4, n_detections) tensor, with bbox details in pixel units (dx, dy, w, h)
            # where dx, dy is a displacement of the box center relative to the cell center

            # Compute bbox centers = cell center + predicted displacement
            bx = xc[n] + temp[0]
            by = yc[n] + temp[1]

            detections[n, :, 0] = bx - 0.5 * temp[2]  # x1
            detections[n, :, 2] = bx + 0.5 * temp[2]  # x2
            detections[n, :, 1] = by - 0.5 * temp[3]  # y1
            detections[n, :, 3] = by + 0.5 * temp[3]  # y2
            detections[n, :, 4] = values[n, :max_detections]

        return detections

    def detect(self, player_feature_map, player_bbox, ball_feature_map):
        # downscale_factor: downscaling factor of the confidence map versus an original image
        player_detections = self.detect_from_map(player_feature_map, self.player_downsampling_factor,
                                                 self.max_player_detections, player_bbox)

        ball_detections = self.detect_from_map(ball_feature_map, self.ball_downsampling_factor,
                                               self.max_ball_detections)

        # Iterate over batch elements and prepare a list with detection results
        output = []
        for player_det, ball_det in zip(player_detections, ball_detections):
            # Filter out detections below the confidence threshold
            player_det = player_det[player_det[..., 4] >= self.player_threshold]
            player_boxes = player_det[..., 0:4]
            player_scores = player_det[..., 4]
            player_labels = torch.tensor([PLAYER_LABEL] * len(player_det), dtype=torch.int64)
            ball_det = ball_det[ball_det[..., 4] >= self.ball_threshold]
            ball_boxes = ball_det[..., 0:4]
            ball_scores = ball_det[..., 4]
            ball_labels = torch.tensor([BALL_LABEL] * len(ball_det), dtype=torch.int64)

            boxes = torch.cat([player_boxes, ball_boxes], dim=0)
            scores = torch.cat([player_scores, ball_scores], dim=0)
            labels = torch.cat([player_labels, ball_labels], dim=0)

            temp = {'boxes': boxes, 'labels': labels, 'scores': scores}
            output.append(temp)

        return output

    def groundtruth_maps(self, boxes, labels, img_shape):
        # Generate ground truth: player location map, player confidence map and ball confidence map
        # targets: List of ground truth player and ball positions
        # img_shape: shape of the input image

        player_loc_t, player_conf_t, ball_conf_t = create_groundtruth_maps(boxes, labels, img_shape,
                                                                           self.player_downsampling_factor,
                                                                           self.ball_downsampling_factor,
                                                                           self.player_delta, self.ball_delta)

        return player_loc_t, player_conf_t, ball_conf_t

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]

        x = self.base_network(x)
        # x must return 2 tensors
        # one (higher spatial resolution) is for ball detection (downsampled by 4)
        # the other (lower spatial resolution) is for players detection (downsampled by 16)
        assert len(x) == 2
        # Same batch size for two tensors
        assert x[0].shape[0] == x[1].shape[0]
        # Same number of channels
        assert x[0].shape[1] == x[1].shape[1]
        # The first has higher spatial resolution then the other
        assert x[0].shape[2] == height // self.ball_downsampling_factor
        assert x[0].shape[3] == width // self.ball_downsampling_factor
        assert x[1].shape[2] == height // self.player_downsampling_factor
        assert x[1].shape[3] == width // self.player_downsampling_factor

        ball_feature_map = self.ball_classifier(x[0])
        player_feature_map = self.player_classifier(x[1])
        player_bbox = self.player_regressor(x[1])

        if self.phase == 'eval' or self.phase == 'detect':
            # In eval and detect mode, convert logits to normalized confidence in [0..1] range
            player_feature_map = self.softmax(player_feature_map)
            ball_feature_map = self.softmax(ball_feature_map)

        if self.phase == 'train' or self.phase == 'eval':
            # Permute dimensions, so channel is the last one (batch_size, h, w, n_channels)
            ball_feature_map = ball_feature_map.permute(0, 2, 3, 1).contiguous()
            player_feature_map = player_feature_map.permute(0, 2, 3, 1).contiguous()
            player_bbox = player_bbox.permute(0, 2, 3, 1).contiguous()
            # loc has shape (n_batch_size, feature_map_size_y, feature_map_size_x, 4)
            # conf has shape (n_batch_size, feature_map_size_y, feature_map_size_x, 2)
            output = (player_bbox, player_feature_map, ball_feature_map)
        elif self.phase == 'detect':
            # Detect bounding boxes
            output = self.detect(player_feature_map, player_bbox, ball_feature_map)

        return output

    def print_summary(self, show_architecture=True):
        # Print network statistics
        if show_architecture:
            print('Base network:')
            print(self.base_network)
            if self.ball_classifier is not None:
                print('Ball classifier:')
                print(self.ball_classifier)
            if self.player_classifier is not None:
                print('Player classifier:')
                print(self.player_classifier)

        ap, tp = count_parameters(self.base_network)
        print('Base network parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.ball_classifier is not None:
            ap, tp = count_parameters(self.ball_classifier)
            print('Ball classifier parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.player_classifier is not None:
            ap, tp = count_parameters(self.player_classifier)
            print('Player classifier parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.player_regressor is not None:
            ap, tp = count_parameters(self.player_regressor)
            print('Player regressor parameters (all/trainable): {}/{}'.format(ap, tp))

        ap, tp = count_parameters(self)
        print('Total (all/trainable): {} / {}'.format(ap, tp))
        print('')


def build_footandball_detector1(phase='train', max_player_detections=100, max_ball_detections=100,
                                player_threshold=0.0, ball_threshold=0.0):
    # phase: 'train' or 'test'
    assert phase in ['train', 'test', 'detect']

    layers, out_channels = fpn.make_modules(fpn.cfg['X'], batch_norm=True)
    # FPN returns 3 tensors for each input: one dowscaled 4 times in each input dimension, the other downscaled 16 times
    # tensor with 2 channels downscaled 4 times is used for ball detection
    # tensor with 2 channels downscaled 16 times is used for the player detection (1 location corresponds to 16x16 pixel block)
    # tensor with 4 channels downscaled 16 times is used for the player bbox regression
    lateral_channels = 32
    i_channels = 32

    base_net = fpn.FPN(layers, out_channels=out_channels, lateral_channels=lateral_channels, return_layers=[1, 3])
    ball_classifier = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1))
    player_classifier = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1))
    player_regressor = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(i_channels, out_channels=4, kernel_size=3, padding=1))
    detector = FootAndBall(phase, base_net, player_regressor=player_regressor, player_classifier=player_classifier,
                           ball_classifier=ball_classifier, ball_threshold=ball_threshold,
                           player_threshold=player_threshold, max_ball_detections=max_ball_detections,
                           max_player_detections=max_player_detections)
    return detector


def model_factory(model_name, phase, max_player_detections=100, max_ball_detections=100, player_threshold=0.0,
                  ball_threshold=0.0):
    if model_name == 'fb1':
        model_fn = build_footandball_detector1
    else:
        print('Model not implemented: {}'.format(model_name))
        raise NotImplementedError

    return model_fn(phase, ball_threshold=ball_threshold, player_threshold=player_threshold,
                    max_ball_detections=max_ball_detections, max_player_detections=max_player_detections)


if __name__ == '__main__':
    net = model_factory('fb2', 'train')
    net.print_summary()

    x = torch.zeros((2, 3, 1024, 1024))
    x = net(x)

    for t in x:
        print(t.shape)

    print('.')
