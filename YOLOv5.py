import torch
import numpy as np
import cv2
import pafy #take videos from yt and pass to model
from time import time


#note: base code developed using: https://www.youtube.com/watch?v=3wdqO_vYMpA&t=0s
class ObjectDetection:
    """
    Implements the YOLO V5 Model on a YT video using OpenCV 
    """

    def __init__(self, url, out_file):
        """
        Initialises the class with the YT Url and the Output File
        :param url: A valid YT URL (prediction is made on this)
        :out_file: A valid output file name.
        :r type: None
        """
        #initilising attributes 
        self._URL = url 
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' #checks if cuda is available and uses it if it is
        print("\n \nDevice Used (if its not cuda gl)", self.device)


    def get_video_from_url(self):
        """
        Generates video streaming object. Frame by frame extraction will be done in order to make predictions
        :return: openCV2 video capture object, with lowest quality frame available for video
        """

        play = pafy.new(self._URL).streams[-1]
        assert play is not None #makes sure play is not none
        return cv2.VideoCapture(play.url) #here you can also give camera input (verify)
    
    def load_model(self):
        """
        Loads YOLO V5 Model from PyTorch
        :return: Train model from PyTorch
        """
        #you can also train your own model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #specity directory, additionally YOLO V5 small model specified (pretrained)
        #if you specify custom, path for weights need to be provided
        return model
    
    def score_frame(self, frame):
        """
        Takes a single frame as input, scores frame using the model
        :param frame: Input frame in numpy/tuple/list format.
        :return: Labels and Coordinates of obj detected by model in that frame
        ::
        """
        #take a frame and do a forward pass
        self.model.to(self.device) #setting device
        frame = [frame]
        results = self.model(frame) #for each frame the boundraies and labels will be stored

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        #keeps coords of boundary boxes so they can be drawn later
        return labels, cord



