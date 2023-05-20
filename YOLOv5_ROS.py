import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
import rosbag

class ObjectDetection:
    """
    Implements the YOLO V5 Model on ROS bag images using OpenCV
    """

    def __init__(self, bag_file, out_file):
        """
        Initializes the class with the ROS bag file and the output file
        :param bag_file: Path to the ROS bag file
        :param out_file: A valid output file name.
        """
        self.bag_file = bag_file
        self.out_file = out_file
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Device Used:", self.device)

    def load_model(self):
        """
        Loads YOLO V5 Model from PyTorch
        :return: Trained model from PyTorch
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input and scores the frame using the model
        :param frame: Input frame in numpy/tuple/list format.
        :return: Labels and coordinates of objects detected by the model in that frame
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given value of label, returns the corresponding string label
        :param x: Numeric label
        :return: Corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a given frame and results as input and overlays bounding boxes and labels on the frame.
        :param results: Contains labels and coordinates predicted by the model on the frame.
        :param frame: Frame that has been scored.
        :return: Frame with bounding boxes and labels overlaid on it
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 0, 255)
                label = self.class_to_label(labels[i])
                confidence = row[4]
                text = f"{label}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def __call__(self):
        """
        This function is called when the class is executed. Reads images from the ROS bag file and outputs the result to a new file.
        """
        bag = rosbag.Bag(self.bag_file)
        bridge = CvBridge()

        # Create a VideoWriter object to save the video
        output_fps = 30  # Set the desired output frames per second (FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(self.out_file, fourcc, output_fps, (1280, 720))  # Update the resolution if needed

        for topic, msg, t in bag.read_messages(topics=['/davis/left/image_raw']):
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.score_frame(cv_image)
            cv_image = self.plot_boxes(results, cv_image)

            cv2.imshow('Object Detection', cv_image)
            cv2.waitKey(1)

            # Write the frame to the video file
            video_writer.write(cv_image)

        cv2.destroyAllWindows()
        bag.close()
        video_writer.release()


# Create an instance of the ObjectDetection class and execute
detection = ObjectDetection("/home/shashank/Downloads/outdoor_day1_data.bag", "output_video.avi")
detection()
