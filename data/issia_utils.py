# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

import numpy as np
import os
import random
from collections import defaultdict
from xml.dom import minidom

import cv2

'''
This module contains procedures to read video sequences and ground truth data from CNR ISSIA dataset.
It contains 6 annotated video sequences from a football game.
Ground truth data contains ball, referee and players  position.  
For more information see: http://www.issia.cnr.it/wp/dataset-cnr-fig/
'''

'''
Part of the ISSIA ground truth data. From *.xgtf file CONFIG section

<descriptor name="BALL" type="OBJECT">
            <attribute dynamic="true" name="BallPos" type="http://lamp.cfar.umd.edu/viperdata#point"/>
            <attribute dynamic="true" name="BallShot" type="http://lamp.cfar.umd.edu/viperdata#bvalue">
                <default>
                    <data:bvalue value="false"/>
                </default>
            </attribute>
	    <attribute dynamic="true" name="PlayerInteractingID" type="http://lamp.cfar.umd.edu/viperdata#dvalue">
		<default>
			<data:dvalue value="-1"/>
		</default>
	    </attribute>
        </descriptor>

'''

# AUXILIARY FUNCTIONS


def _dist(x1, x2):
    # Euclidean distance between two points
    return np.sqrt((float(x1[0])-float(x2[0]))**2 + (float(x1[1])-float(x2[1]))**2)


def _parse_framespan(framespan):
    '''
    Auxiliary function to parse frame span value
    :param framespan: string in "start_frame:end_frame" format, e.g. "1182:1185"
    :return: a tuple of int (start_frame, end_frame)
    '''
    # Parse framespan, format is: framespan="1182:1185"
    pos = framespan.find(':')
    if pos < 1:
        assert False, 'Incorrect framespan value: ' + framespan

    start_frame = int(framespan[:pos])
    end_frame = int(framespan[pos + 1:])
    return start_frame, end_frame


def _load_groundtruth(filepath):
    '''
    Load ground truth data from XML file (ISSIA dataset)
    :param filepath: Path to the ground truth XML file
    :return: Dictionary with ISSIA ground truth data. The dictionary has the following elements:
             gt['BallPos']              - ball positions at each frame
             gt['BallShot']             - ball shot events
             gt['PlayerInteractingID']  - ID of the player interacting with a ball
             gt['Person']               - players bounding boxes
    '''

    assert os.path.isfile(filepath)
    xmldoc = minidom.parse(filepath)

    # Processs information from file section. Extract number of frames (NUMFRAMES) value.
    itemlist = xmldoc.getElementsByTagName('file')
    # There should be exactly 1 file element in a groundtruth file
    assert len(itemlist) == 1

    num_frames = None

    for e in itemlist[0].getElementsByTagName('attribute'):
        if e.attributes['name'].value =='NUMFRAMES':
            values = e.getElementsByTagName('data:dvalue')
            # There should be only one data:dvalue node
            assert len(values) == 1
            num_frames = values[0].attributes['value'].value
            break

    if num_frames is None:
        assert False, 'NUMFRAMES not definedin XML file.'

    print('Number of frames = ' + str(num_frames))

    # Processs information from data section. Extract ball positions

    # Dictionary to hold ground truth values
    gt ={}
    # List of ball position on each frame from the sequence
    gt['BallPos'] = []
    # Indicates if the ball was shot on each frame from the sequence
    gt['BallShot'] = []
    # ID of player interacting with the ball on each frame from the sequence
    gt['PlayerInteractingID']= []
    # Dictionary storing list of bounding boxes of each person for each frame from the sequence
    gt['Person'] = defaultdict(list)

    itemlist = xmldoc.getElementsByTagName('object')
    # This returns multiple object elements (BALL, Person)

    for e in itemlist:
        assert 'name' in e.attributes
        if e.attributes['name'].value == 'BALL':
            ball_attributes = e.getElementsByTagName('attribute')
            # Valid ball attributes are: BallPos, BallShot, PlayerInteractingID

            for elem in ball_attributes:
                if elem.attributes['name'].value == 'BallPos':
                    for e in elem.getElementsByTagName('data:point'):
                        # <data:point framespan='1182:1182' x='91' y='443'/>
                        assert 'framespan' in e.attributes
                        assert 'x' in e.attributes
                        assert 'y' in e.attributes

                        framespan = e.attributes['framespan'].value
                        x = int(e.attributes['x'].value)
                        y = int(e.attributes['y'].value)
                        start_frame, end_frame = _parse_framespan(framespan)
                        gt['BallPos'].append((start_frame, end_frame, x, y))

                elif elem.attributes['name'].value == 'BallShot':
                    for e in elem.getElementsByTagName('data:bvalue'):
                        # <data:bvalue framespan="1182:1182" value="false"/>
                        assert 'framespan' in e.attributes
                        assert 'value' in e.attributes

                        framespan = e.attributes['framespan'].value
                        value = (e.attributes['value'].value == 'true')
                        start_frame, end_frame = _parse_framespan(framespan)
                        gt['BallShot'].append((start_frame, end_frame, value))

                elif elem.attributes['name'].value == 'PlayerInteractingID':
                    for e in elem.getElementsByTagName('data:dvalue'):
                        # <data:dvalue framespan="1182:1182" value="-1"/>
                        assert 'framespan' in e.attributes
                        assert 'value' in e.attributes

                        framespan = e.attributes['framespan'].value
                        value = int(e.attributes['value'].value)
                        start_frame, end_frame = _parse_framespan(framespan)
                        gt['PlayerInteractingID'].append((start_frame, end_frame, value))

                else:
                    assert False, "Unexpected attribute: " + elem.attributes['name'].value

        elif e.attributes['name'].value == 'Person':
            person_id = e.attributes['id'].value
            person_attributes = e.getElementsByTagName('attribute')
            for elem in person_attributes:
                if elem.attributes['name'].value == 'LOCATION':
                    for e in elem.getElementsByTagName('data:bbox'):
                        # <data:point framespan='1182:1182' x='91' y='443'/>
                        assert 'framespan' in e.attributes
                        assert 'height' in e.attributes
                        assert 'width' in e.attributes
                        assert 'x' in e.attributes
                        assert 'y' in e.attributes

                        framespan = e.attributes['framespan'].value
                        height = int(e.attributes['height'].value)
                        width = int(e.attributes['width'].value)
                        x = int(e.attributes['x'].value)
                        y = int(e.attributes['y'].value)
                        start_frame, end_frame = _parse_framespan(framespan)
                        gt['Person'][person_id].append((start_frame, end_frame, height, width, x, y))

                else:
                    assert False, "Unexpected attribute: " + elem.attributes['name'].value

    return gt


def _create_annotations(gt, camera_id, frame_shape):
    '''
    Convert ground truth from ISSIA dataset to SequenceAnnotations object
    Camera id and frame shape is needed to rectify the ball position (as for some strange reason actual ball position
    in ISSIA dataset is shifted/reversed for some sequences)
    For each frame we have:
    - list of ball positions (in ISSIA dataset there's only one ball, but in other datasets we can have more)
    - boolean flag showing if a ball was shot
    - list of ids of players interacting with the ball
    - list of bounding boxes of players visible in this frame

    :param gt: dictionary with ISSIA ground truth data returned by load_groundtruth function
    :return: SequenceAnnotations object with ground truth data
    '''

    annotations = SequenceAnnotations()

    # In ISSIA dataset there's a discrepancy between frame number in the sequence and ground truth data
    # We need to add delta = 8 to the ground truth data, to get the real frame number (frame numbers start from 1)
    # delta = -8 works well for all sequences
    delta = -8

    for (start_frame, end_frame, x, y) in gt['BallPos']:
        for i in range(start_frame, end_frame+1):
            if camera_id == 2 or camera_id == 6:
                # For some reason ball coordinates for camera 2 and 6 have reversed in x-coordinate
                x = frame_shape[1] - x
            annotations.ball_pos[i+delta].append((x, y))

    for (start_frame, end_frame, value) in gt['BallShot']:
        if value:
            for i in range(start_frame, end_frame+1):
                annotations.ball_shot[i+delta] = True

    for (start_frame, end_frame, value) in gt['PlayerInteractingID']:
        if value > -1:
            for i in range(start_frame, end_frame+1):
                annotations.interacting_player[i+delta].append(value)

    for player in gt['Person']:
        for (start_frame, end_frame, height, width, x, y) in gt['Person'][player]:
            assert start_frame <= end_frame
            for i in range(start_frame, end_frame+1):
                annotations.persons[i+delta].append((player, height, width, x, y))

    return annotations


def _ball_detection_stats(ball_pos, gt_ball_pos, tolerance):
    '''
    Compute ball detection stats for the single frame
    :param ball_pos: A list of detected ball positions. Multiple balls are possible.
    :param gt_ball_poss: A list of ground truth ball positions. Multiple balls are possible.
    :param tolerance: tolerance for ball centre tolerance in pixels
    :return: A tuple (precision, recall, number of correctly_classified frames)
    '''

    # True positives, false positives and false negatives
    tp = 0
    fp = 0
    fn = 0

    # Another count of true positives based on enumerating ground truth detections
    # If tp != tp1 this means that more than one ball was detected for one ground truth ball
    tp1 = 0

    # For each detected ball, check if it corresponds to a ground truth ball
    for e in ball_pos:
        # Verify if it's a true positive or a false positive
        hit = False
        for gt_e in gt_ball_pos:
            if _dist(e, gt_e) <= tolerance:
                # Matching ground truth ball position
                hit = True
                break

        if hit:
            # Ball correctly detected - true positive
            tp += 1
        else:
            # Ball incorrectly detected - false positive
            fp += 1

    # For each ground truth  ball, check if it was detected
    for gt_e in gt_ball_pos:
        # Verify if it's a false negative
        hit = False
        for e in ball_pos:
            if _dist(e, gt_e) <= tolerance:
                # Matching ground truth ball position
                hit = True
                break

        if hit:
            tp1 += 1
        else:
            # Ball not detected - false negative
            fn += 1

    precision = None
    recall = None

    if tp+fp > 0:
        precision = tp/(tp+fp)

    if tp+fn > 0:
        recall = tp/(tp+fn)

    # Frame was correctly classified if there were no false positives and false negatives
    correctly_classified = (fp == 0 and fn == 0)

    return precision, recall, correctly_classified


def _annotate_frame(frame, frame_id, annotations, color=(0, 0, 255)):
    '''
    Visualize annotations. Draw ball position and players' bounding boxes.
    :param frame: input video frame
    :param frame_id: id of the video frame
    :param annotations: SequenceAnnotations object
    :param color: color
    :return: annotated video frame (with ball position and players' bounding boxes)
    '''
    # Ball position
    for (x,y) in annotations.ball_pos[frame_id]:
        if x > -1 and y > -1:
            cv2.circle(frame, (x, y), 10, color, -1)

    #Ball shot
    #if annotations.ball_shot[frame_id]:
    #   print('Ball shot...')

    # Interacting player
    #for value in annotations.interacting_player[frame_id]:
    #    print('Interacting player: ' + str(value))

    # Person bounding boxes
    for (player, height, width, x, y) in annotations.persons[frame_id]:
        cv2.rectangle(frame, (x, y), (x+width, y+height), color)

    return frame


# Functions to be exported


class SequenceAnnotations:
    '''
    Class for storing annotations for the video sequence
    '''
    def __init__(self):
        # ball_pos contains list of ball positions (x,y) on each frame; multiple balls per frame are possible
        self.ball_pos = defaultdict(list)
        # ball_shot contains a boolean flag if a ball was shot for each frame
        self.ball_shot = defaultdict(bool)
        # interacting_player contains a list of players interacting with the ball on each frame
        self.interacting_player = defaultdict(list)
        # persons contains a list of bounding boxes for players visible on the frame
        self.persons = defaultdict(list)


def open_issia_sequence(camera_id, dataset_path):
    '''
    Open the video sequence from the camera: camera_id
    :param camera_id: number of the ISSIA sequence (between 1 and 6)
    :return: VideoStream
    '''
    assert (camera_id >= 1) and (camera_id <= 6)

    dataset_path = os.path.expanduser(dataset_path)
    camera_file = 'filmrole' + str(camera_id) + str('.avi')
    camera_filepath = os.path.join(dataset_path, camera_file)

    assert os.path.isfile(camera_filepath)
    sequence = cv2.VideoCapture(camera_filepath)
    return sequence


def read_issia_ground_truth(camera_id, dataset_path):
    '''
    Read ground truth from the ISSIA dataset correpsoinding to camera: camera_id
    :param camera_id: number of teh sequence (between 1 and 6)
    :return: SequenceAnnotations object
    '''
    assert (camera_id >= 1) and (camera_id <= 6)

    dataset_path = os.path.expanduser(dataset_path)
    annotation_path = os.path.join(dataset_path, 'Annotation Files')
    annotation_file = 'Film Role-0 ID-' + str(camera_id) + ' T-0 m00s00-026-m00s01-020.xgtf'
    annotation_filepath = os.path.join(annotation_path, annotation_file)

    gt = _load_groundtruth(annotation_filepath)

    # Read the first frame from the video stream and close it
    # This is needed to get frame size, so ISSIA ground truth can be rectified
    sequence = open_issia_sequence(camera_id, dataset_path=dataset_path)
    ret, frame = sequence.read()
    sequence.release()

    annotations = _create_annotations(gt, camera_id, frame.shape)

    return annotations


def evaluate_ball_detection_results(annotations, gt_annotations, tolerance):
    '''
    Evaluate ball detection performance
    :param annotations: SequenceAnnotations object with ball detection results to be evaluated
    :param gt_annotations: SequenceAnnotations object with ground truth annotations
    :param tolerance: tolerance in pixels for ball centre detection
    :return: A tuple (precision, recall, percent of correctly_classified frames)
    '''

    frame_stats = []

    start_frame = min(gt_annotations.ball_pos)
    end_frame = max(gt_annotations.ball_pos)

    for i in range(start_frame, end_frame):
        ball_pos = annotations.ball_pos[i]
        gt_ball_pos = gt_annotations.ball_pos[i]
        frame_stats.append(_ball_detection_stats(ball_pos, gt_ball_pos, tolerance))

    percent_correctly_classified_frames = sum([c for (_,_,c) in frame_stats])/len(frame_stats)
    temp = [p for (p, _, _) in frame_stats if p is not None]
    avg_precision = sum(temp)/len(temp)

    temp = [r for (_, r, _) in frame_stats if r is not None]
    avg_recall = sum(temp) / len(temp)

    return avg_precision, avg_recall, percent_correctly_classified_frames


def visualize_detection_results(camera_id, dataset_path, gt_annotations=None, annotations=None):
    '''
    Visualize ground truth annotations (in blue) and detected annotations (in red)
    :param camera_id: ID of the ISSIA video sequence
    :param gt_annotations: SequenceAnnotations object with ground truth annotations
    :param annotations: SequenceAnnotations object with detected annotations. If None only ground truth annotations are
                        shown
    :return:
    '''
    sequence = open_issia_sequence(camera_id, dataset_path)
    count_frames = -1
    while (sequence.isOpened()):
        ret, frame = sequence.read()
        count_frames += 1

        if not ret:
            # End of sequence
            break

        if not gt_annotations is None:
            frame = _annotate_frame(frame, count_frames, gt_annotations, color=(0, 0, 255))

        if not annotations is None:
            frame = _annotate_frame(frame, count_frames, annotations, color=(255, 0, 0))

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey()

    sequence.release()
    cv2.destroyAllWindows()


def extract_frames(dataset_path, camera_id, frames_path):
    # Extract frames from the sequence
    print('Extracting sequence: ' + str(camera_id) + ' from: ' + dataset_path + ' ...')
    sequence = open_issia_sequence(camera_id, dataset_path)
    count_frames = -1
    while (sequence.isOpened()):
        ret, frame = sequence.read()
        count_frames += 1

        if not ret:
            # End of sequence
            break

        file_path = os.path.join(frames_path, str(count_frames) + '.png')
        cv2.imwrite(file_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    sequence.release()
    cv2.destroyAllWindows()
    print('Done')


if __name__ == '__main__':
    # Example to demonstrate usage of module procedures
    print("OpenCV version: " + str(cv2.__version__))

    # Read ISSIA sequence and visualize ground truth
    # ISSIA dataset can be downloaded from http://www.issia.cnr.it/wp/dataset-cnr-fig/
    # Camera ids are between 1 and 6
    dataset_path = '/media/jacek/312b3cfe-6e0b-4a43-9b11-d163c1d5d5ad/data/issia'
    camera_id = 1
    sequence = open_issia_sequence(camera_id, dataset_path)

    # Read annotations included in the dataset
    gt_annotations = read_issia_ground_truth(camera_id, dataset_path)

    # Show annotated video sequence
    visualize_detection_results(camera_id, dataset_path, gt_annotations=gt_annotations)

    # Ball detection in pixels performance
    # This should return all ones as we evaluate the performance on ground truth data
    tolerance = 3
    avg_precision, avg_recall, percent_correctly_classified_frames = evaluate_ball_detection_results(gt_annotations,
                                                                                                     gt_annotations,
                                                                                                     tolerance=tolerance)

    print('Avg. precision = ' + str(avg_precision))
    print('Avg. recall = ' + str(avg_recall))
    print('Percent of correctly classified frames = ' + str(percent_correctly_classified_frames))
