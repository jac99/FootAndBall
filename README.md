

## FootAndBall: *Integrated player and ball detector*
Created by [Jacek Komorowski](mailto:jacek.komorowski@pw.edu.pl),
Grzegorz Kurzejamski and Grzegorz Sarwas
at <a href="https://sagsport.com/?lang=en" target="_blank">Sport Algorithmics and Gaming</a>

**System overview:**
<p align="center"> <img src="visualization/demo.gif" width="100%"> </p>

### Abstract
The paper describes a deep neural network-based detector dedicated for ball and players detection in high resolution, long shot, video recordings of soccer matches. The detector, dubbed FootAndBall, has an efficient fully
convolutional architecture and can operate on input video stream with an arbitrary resolution. It produces ball
confidence map encoding the position of the detected ball, player confidence map and player bounding boxes
tensor encoding players’ positions and bounding boxes. The network uses Feature Pyramid Network desing
pattern, where lower level features with higher spatial resolution are combined with higher level features with
bigger receptive field. This improves discriminability of small objects (the ball) as larger visual context around
the object of interest is taken into account for the classification. Due to its specialized design, the network has
two orders of magnitude less parameters than a generic deep neural network-based object detector, such as
SSD or YOLO. This allows real-time processing of high resolution input video stream.

### Environment and Dependencies
* Ubuntu (18.04 or 20.04) with CUDA (10.1 or 10.2) + PyTorch (1.6 or above)

Other dependencies include:
* scipy
* opencv
* kornia
* PIL


### Datasets
Our model is mainly trained 
<a href="https://drive.google.com/open?id=1l4jGKES04iLq9b0CPlvIevaAN8YDZ2ne" target="_blank">using ISSIA-CNR Soccer dataset</a> 
(camera 1, 2, 3 and 3) 
and
<a href="http://www.cs.ubc.ca/labs/lci/datasets/SoccerPlayerDetection_bmvc17_v1.zip" target="_blank">SoccerPlayerDetection_bmvc17_v1 dataset</a>
. 

### Training
To train our network, edit config.txt and set paths to training datasets.
Then, run:
    python train_detector.py --config config.txt


### Testing
The pre-trained model *model_20201019_1416_final.pth* is saved in `models/` folder.
To run the trained model use the following command:

    python run_detector --path datasets/issia/filmrole5.avi --weights models/model_20201019_1416_final.pth --out_video out_video.avi --device <cpu or cuda>

This will create a video with name given by *out_video* parameter with bounding boxes around the ball and 
players position. Detection confidence level (in 0..1 range) will be displayed above each bounding box. 


### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{DBLP:conf/visapp/KomorowskiKS20,
      author    = {Jacek Komorowski and
                   Grzegorz Kurzejamski and
                   Grzegorz Sarwas},
      editor    = {Giovanni Maria Farinella and
                   Petia Radeva and
                   Jos{\'{e}} Braz},
      title     = {FootAndBall: Integrated Player and Ball Detector},
      booktitle = {Proceedings of the 15th International Joint Conference on Computer
                   Vision, Imaging and Computer Graphics Theory and Applications, {VISIGRAPP}
                   2020, Volume 5: VISAPP, Valletta, Malta, February 27-29, 2020},
      pages     = {47--56},
      publisher = {{SCITEPRESS}},
      year      = {2020},
      url       = {https://doi.org/10.5220/0008916000470056},
      doi       = {10.5220/0008916000470056}
    }
    

### License
Our code is released under the MIT License (see LICENSE file for details).

