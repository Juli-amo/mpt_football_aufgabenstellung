import numpy as np
import cv2

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!
        self.previous_frame = None  # Store the previous frame for optical flow calculation
        self.current_frame = None  # Store the current frame for processing

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

        # Grayscale conversion of the current frame
        if "image" not in data or data["image"] is None:
            return {"opticalFlow": np.array([[0.0, 0.0]])}
        else:
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # If this is the first frame, we cannot calculate motion yet
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return {"opticalFlow": np.array([[0.0, 0.0]])}

        # also eigentlich müssen die Dinger aus dem Trackern kommen, aber da die noch nicht da sind, kommt hier erstmal diese funtkion
        prev_features_tracked = cv2.goodFeaturesToTrack(previous_frame,
                                           maxCorners=100,
                                           qualityLevel=0.3,
                                           minDistance=7,
                                           blockSize=7)

        if prev_features_tracked is None:
            self.previous_frame = current_frame.copy()
            return {"opticalFlow": np.array([[0.0, 0.0]])}
        
        # Actually calculate the optical flow using Lucas-Kanade method
        next_features, status = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, prev_features_tracked, None)

        # Filter out bad points
        good_old = prev_features_tracked[status == 1]
        good_new = next_features[status == 1]

        if len(good_old) == 0 or len(good_new) == 0:
            # Update previous frame for next iteration
            self.previous_frame = current_frame.copy()
            return {"opticalFlow": np.array([[0.0, 0.0]])}
        
        # Calculate motion vectors
        motion_vectors = good_new - good_old
        
        # Calculate average motion vector
        mean_motion = np.mean(motion_vectors, axis=0)
        
        # Update the previous frame to the current one for the next step
        previous_frame = current_frame.copy()

        # Rückgabe des Motion Vectors
        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {
           "opticalFlow": np.array(mean_motion)
        }