import torch
import numpy as np
import cv2
import pafy #take videos from yt and pass to model
from time import time

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

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    


