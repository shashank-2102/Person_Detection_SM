import torch
import numpy as np
import cv2
import pafy #take videos from yt and pass to model
from time import time
import rosbag
import cv2
from cv_bridge import CvBridge

#note: base code developed using: https://www.youtube.com/watch?v=3wdqO_vYMpA&t=0s

class ObjectDetection:
    """
    Implements the YOLO V5 Model on a YT video, webcam or local file using OpenCV 
    """

    def __init__(self, url, inp_typ, out_file):
        """
        Initialises the class with the YT Url and the Output File
        :param url: A valid YT URL OR Local file location
        :paral inp_typ: User defined either 'Webcam', 'Local' or 'YT'
        :out_file: A valid output file name.
        :r type: None
        """
        #initilising attributes
        self.input_t = inp_typ 
        self._URL = url  #can be YT url or local file path
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu :(' #checks if cuda is available and uses it if it is
        print("\n \nDevice Used (if its not cuda gl)", self.device)

    def get_video_from_url(self):
        """
        Generates video streaming object. Frame by frame extraction will be done in order to make predictions
        :return: openCV2 video capture object, with lowest quality frame available for video
        """
        #distingish between local file and YT input
        
        
        if self.input_t == "Webcam":
            print("Opening Webcam")
            cap = cv2.VideoCapture(0)
            # Set resolution of input frames to 640x480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Set frame rate of input frames to 30 frames per second
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        
        elif self.input_t == "Local":
            print("Loading local video file")
            input_file = self._URL #test for mp4
            cap = cv2.VideoCapture(input_file)
            # Set resolution of input frames to 640x480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #doesnt seem to work yet, check

            # Set frame rate of input rames to 30 frames per second
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap

        elif self.input_t == "YT":
            print("Loading YT Video")
            play = pafy.new(self._URL).streams[-1]
            input_file = play.url
            cap = cv2.VideoCapture(input_file)
            # Set resolution of input frames to 640x480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5)

            # Set frame rate of input frames to 30 frames per second
            cap.set(cv2.CAP_PROP_FPS, 100)
            return cap
        
        elif self.input_t == "Rosbag":
            print("Loading Rosbag file")
            
            input_file = self._URL
            return input_file
        

    
    def load_model(self):
        """
        Loads YOLO V5 Model from PyTorch
        :return: Train model from PyTorch
        """
        #you can also train your own model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
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
        #keeps labels/coords of boundary boxes so they can be drawn later
        #take all val of first col, and last index in [:, -1]

    

        return labels, cord
    
    def class_to_label(self, x):
        """
        For given value of label, return string label
        :param x: numeric label
        :return: corresponding string label
        :r type: string
        """
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        """
        Takes a given frame and results as input and then overlays bounding boxes and labels on the frame.
        :param results: Contains labels and coords predicted by model on frame.
        :param frame: Frame that has been scored.
        :return: Frame with bounding boxes and labels overlayed on it
        """
        labels, cord = results
        n = len(labels) #number of detected labels
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n): #running through all the detections
            row = cord[i]
            if row[4]>=0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr =  (0, 0, 255) #colour of boundary box, currently red
                label = self.class_to_label(labels[i])
                confidence = row[4]
                text = f"{label}: {confidence:.2f}" #label and confidence text to be shown
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2) #draw rectangle around object
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2) #displaying correspoding label
        return frame 

    def __call__(self):
        """
        This function is called when the class is executed. Runs loop to read video frame by frame and outputs the result to a new file
        :return: void
        """
        if self.input_t=="Rosbag":
            bag = rosbag.Bag(self._URL)
            bridge = CvBridge()
            x_shape, y_shape = 1280, 720 #output resolution

        else:
            player = self.get_video_from_url()
            assert player.isOpened()
            x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT)) #output resolution

        four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(self.out_file, four_cc, 30, (x_shape, y_shape))

        if self.input_t=="Rosbag":
            for topic, msg, t in bag.read_messages(topics=['/davis/left/image_raw']):
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                results = self.score_frame(cv_image)
                cv_image = self.plot_boxes(results, cv_image)

                cv2.imshow('Object Detection', cv_image)
                cv2.waitKey(1)

                # Write the frame to the video file
                out.write(cv_image)

        

        else:
            while True: #as long as you have frames in video
                start_time = time() #timer
                ret, frame = player.read() #load frame from video
                if not ret:
                    break

                results = self.score_frame(frame) #get results
                frame = self.plot_boxes(results, frame) #plot boxes
                
                # Display the frame with bounding boxes and labels in real-time
                cv2.imshow('Object Detection', frame)
                cv2.waitKey(1)  # Wait for a key event (1 millisecond delay)

                # Check for 'q' key press to exit the video processing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                end_time = time()
                fps = 1/np.round(end_time-start_time, 3) #calculate fps
                print(f"FPS:{fps}")
                print(x_shape, y_shape)
                out.write(frame)
            # Release the video capture and close the window
    
        cv2.destroyAllWindows()
        bag.close()
        out.release()


#create new obj and execute
#give video url and output file name

detection = ObjectDetection("/home/shashank/Downloads/outdoor_day1_data.bag", "Rosbag", "video_t7.avi")
detection()
#choose between 'Local', 'Webcam' and 'YT' for input
#either give URL or path for YT and Local respectively


