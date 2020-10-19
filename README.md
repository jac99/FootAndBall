

## FootAndBall: *Integrated player and ball detector*
**[FootAndBall: Integrated player and ball detector](https://www.scitepress.org/Link.aspx?doi=10.5220/0008916000470056)** VISAPP 2020, Valletta, Malta. 

Created by [Jacek Komorowski](mailto:jacek.komorowski@pw.edu.pl),
Grzegorz Kurzejamski and Grzegorz Sarwas
at <a href="https://sagsport.com/?lang=en" target="_blank">Sport Algorithmics and Gaming</a>

<p align="center"> <img src="images/demo.gif" width="100%"> </p>

### Abstract
The paper describes a deep neural network-based detector dedicated for ball and players detection in high resolution, 
long shot, video recordings of soccer matches. The detector, dubbed **FootAndBall**, has an efficient fully
convolutional architecture and can operate on input video stream with an arbitrary resolution. It produces ball
confidence map encoding the position of the detected ball, player confidence map and player bounding boxes
tensor encoding players’ positions and bounding boxes. The network uses Feature Pyramid Network desing
pattern, where lower level features with higher spatial resolution are combined with higher level features with
bigger receptive field. This improves discriminability of small objects (the ball) as larger visual context around
the object of interest is taken into account for the classification. Due to its specialized design, the network has
two orders of magnitude less parameters than a generic deep neural network-based object detector, such as
SSD or YOLO. This allows real-time processing of high resolution input video stream.

### System overview
<p align="center"> <img src="images/overview.png" width="100%"> </p>

### Citation
If you find our work useful, please consider citing:

    @conference{visapp20,
    author={Jacek Komorowski. and Grzegorz Kurzejamski. and Grzegorz Sarwas.},
    title={FootAndBall: Integrated Player and Ball Detector},
    booktitle={Proceedings of the 15th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,},
    year={2020},
    pages={47-56},
    publisher={SciTePress},
    organization={INSTICC},
    doi={10.5220/0008916000470056},
    isbn={978-989-758-402-2},
    }
    
### Environment and Dependencies
Code was tested using Python 3.6 with PyTorch 1.6 on Ubuntu 18.04 with CUDA 10.2.
Other dependencies include:
* Python (1.6 or above)
* scipy
* opencv-python
* kornia
* PIL


### Datasets
Our model is trained using
<a href="https://drive.google.com/file/d/1Pj6syLRShNQWQaunJmAZttUw2jDh8L_f/view?usp=sharing" target="_blank">ISSIA-CNR Soccer dataset</a> 
and
<a href="http://www.cs.ubc.ca/labs/lci/datasets/SoccerPlayerDetection_bmvc17_v1.zip" target="_blank">SoccerPlayerDetection_bmvc17_v1 dataset</a>
<a href="https://drive.google.com/file/d/1ctJojwDaWtHEAeDmB-AwEcO3apqT-O-9/view?usp=sharing" target="_blank">(alternative link)</a>
. 

### Training
To train **FootAndBall** detector, edit `config.txt` and set paths to ISSIA-CNR Soccer and SoccerPlayerDetection (optionally) training datasets.
Then, run:
    
    python train_detector.py --config config.txt


### Testing
The pre-trained model `model_20201019_1416_final.pth` is saved in `models/` folder.
The model was trained with ISSIA-CNR dataset (cameras 1,2,3,4) and SoccerPlayerDetection dataset (set 1).
To run the trained model use the following command:

    python run_detector --path datasets/issia/filmrole5.avi --weights models/model_20201019_1416_final.pth --out_video out_video.avi --device <cpu|cuda>

This will create a video with name given by *out_video* parameter with bounding boxes around the ball and 
players position. Detection confidence level (in 0..1 range) will be displayed above each bounding box. 
Exemplary videos with detection results on ISSIA-CNR Soccer dataset can ve downloaded here:
[camera 5](images/results5.mp4)
[camera 6](images/results6.mp4)
. 

### License
Our code is released under the MIT License (see LICENSE file for details).

