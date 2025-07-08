import cv2 
import numpy as np

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!
        self.prev_gray = None  # Previous grayscale image
        self.prev_positions = None  # Previous tracked object positions

    def start(self, data):
        # Start up procedure of the module - unnecessary for this module, but required for interface consistency
        self.prev_gray = None
        self.prev_positions = None

    def stop(self, data):
        # Shut down procedure of the module - unnecessary for this module, but required for interface consistency
        self.prev_gray = None
        self.prev_positions = None

    def step(self, data):
        # Get current image and tracks from tracker
        image = data["image"]
        tracks = data.get("tracks", np.array([]).reshape(0, 4))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Initialize optical flow to zero for first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            return {"opticalFlow": np.array([0.0, 0.0], dtype=np.float32)}
        
        # If no tracks are available, return zero
        if len(tracks) == 0:
            self.prev_gray = gray
            return {"opticalFlow": np.array([0.0, 0.0], dtype=np.float32)}
        
        # Extract center points from tracks [x, y, w, h] -> [x, y]
        current_positions = tracks[:, :2].reshape(-1, 1, 2).astype(np.float32)
        
        # Get previous positions from previous frame
        if hasattr(self, 'prev_positions') and self.prev_positions is not None:
            if len(self.prev_positions) == len(current_positions):
                # Displacement between previous and current track positions
                displacements = current_positions.reshape(-1, 2) - self.prev_positions.reshape(-1, 2)
                # Average displacement represents camera motion
                mean_flow = -np.mean(displacements, axis=0) 
            else:
                mean_flow = np.array([0.0, 0.0])
        else:
            mean_flow = np.array([0.0, 0.0])
        
        # Store current positions for next frame
        self.prev_positions = current_positions.copy()
        self.prev_gray = gray

        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {
           "opticalFlow": mean_flow.astype(np.float32)
        }

