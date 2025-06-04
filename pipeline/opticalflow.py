import numpy as np
import cv2

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!

    def start(self, data):
        # TODO: Implement start up procedure of the module
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        # TODO: Implement processing of a single frame
        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image. 
        current_frame = data["image"]
        previous_frame = None
        motion_vector = {[0, 0]} 

        if previous_frame is None:  # If there is no previous frame, there is no motion to calculate
            previous_frame = current_frame
            return {"opticalFlow": np.array([[0.0, 0.0]])}

        # also eigentlich brauche ich noch die dinge die getrckt werden sollen vom Tracker deswegen idk

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {
           "opticalFlow": None
        }