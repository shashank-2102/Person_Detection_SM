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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n \nDevice Used (if its not cuda gl)", self.device)


    


