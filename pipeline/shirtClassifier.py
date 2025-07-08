import numpy as np
import cv2

class ShirtClassifier:
    def __init__(self, update_rate=0.1, use_hsv=True):
        #Name identifier for the classifier
        self.name = "Shirt Classifier"
        #Flag indicating whether initial team colors have been determined
        self.initialized = False
        #Cached team colors (as HSV or BGR arrays) for team A and B
        self.teamA_color = None
        self.teamB_color = None
        #Rate at which to update the running average of team colors
        self.update_rate = update_rate  #running average rate
        #Flag to choose color space: HSV for robustness to lighting, else BGR
        self.use_hsv = use_hsv

    def start(self, data):
        """
        Called at the beginning of a new video/run: resets internal state.
        """
        self.initialized = False
        self.teamA_color = None
        self.teamB_color = None

    def stop(self, data):
        """
        Called at the end of processing. Currently unused.
        """
        pass

    def step(self, data):
        """
        Main per-frame processing method. Returns team colors and assignments.

        Args:
            data (dict): contains keys:
                - 'image': current frame (H x W x 3 BGR image)
                - 'tracks': list of bounding boxes [x, y, w, h]
                - 'trackClasses': list of class labels per track (2 = player)

        Returns:
            dict with:
                - 'teamAColor', 'teamBColor': output colors as BGR tuples
                - 'teamClasses': list of 0/1/2 assignments per track
        """
        #Retrieve input data
        image = data.get("image")
        tracks = data.get("tracks", [])
        track_classes = data.get("trackClasses", [])

        #If no players detected, return default zeros
        if len(tracks) == 0 or len(track_classes) == 0:
            return {
                "teamAColor": (0, 0, 0),
                "teamBColor": (0, 0, 0),
                "teamClasses": []
            }

        #Extract mean colors from player crops
        colors = []     #list of mean color vectors
        indices = []    #corresponding indices in original track list
        for idx, (track, cls) in enumerate(zip(tracks, track_classes)):
            #Only process if this track is classified as a player (class 2)
            if cls == 2:
                #Unpack and convert to ints
                x, y, w, h = track.astype(int)
                #Compute crop coordinates (clamped to image borders)
                x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
                x2, y2 = min(x + w // 2, image.shape[1] - 1), min(y + h // 2, image.shape[0] - 1)
                #Crop the player region
                crop = image[y1:y2, x1:x2]
                #Only if crop is non-empty
                if crop.size > 0:
                    if self.use_hsv:
                        #Convert to HSV and compute mean per channel
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mean = np.mean(hsv.reshape(-1, 3), axis=0)
                    else:
                        #Compute mean in BGR space
                        mean = np.mean(crop.reshape(-1, 3), axis=0)
                    colors.append(mean)
                    indices.append(idx)

        #If fewer than 2 players, cannot cluster: fallback to default assignments
        if len(colors) < 2:
            #fallback: assign no teams
            team_classes = [0 if c != 2 else 1 for c in track_classes]
            return {
                #Use cached colors if available, else zeros
                "teamAColor": tuple(int(c) for c in (self.teamA_color or (0,0,0))),
                "teamBColor": tuple(int(c) for c in (self.teamB_color or (0,0,0))),
                "teamClasses": team_classes
            }
        #Convert list to array for vector ops
        colors = np.array(colors)

        #initialization on first frame
        if not self.initialized:
            #simple 2-means clustering
            #Start with the first two samples
            c0, c1 = colors[0], colors[1]
            #Perform 10 iterations of mean update
            for _ in range(10):
                #Compute distances to cluster centers
                d0 = np.linalg.norm(colors - c0, axis=1)
                d1 = np.linalg.norm(colors - c1, axis=1)
                #Assign each sample to the closer center (0 or 1)
                lbls = (d1 < d0).astype(int)
                #Recompute centers if cluster non-empty
                if np.any(lbls == 0): c0 = np.mean(colors[lbls == 0], axis=0)
                if np.any(lbls == 1): c1 = np.mean(colors[lbls == 1], axis=0)
            #set/cache resulting team colors
            self.teamA_color = c0
            self.teamB_color = c1
            self.initialized = True
            labels = lbls
        else:
            #Subsequent frames: classify by distance to cached centers 
            dA = np.linalg.norm(colors - self.teamA_color, axis=1)
            dB = np.linalg.norm(colors - self.teamB_color, axis=1)
            #Label 0 if closer to A, 1 if closer to B
            labels = (dB < dA).astype(int)
            #Adapt centers via running average to handle lighting changes
            #Compute new means of samples assigned to each team
            newA = np.mean(colors[labels == 0], axis=0) if np.any(labels==0) else None
            newB = np.mean(colors[labels == 1], axis=0) if np.any(labels==1) else None
            #Update cached colors
            if newA is not None:
                self.teamA_color = (1 - self.update_rate) * self.teamA_color + self.update_rate * newA
            if newB is not None:
                self.teamB_color = (1 - self.update_rate) * self.teamB_color + self.update_rate * newB

        #Build the output teamClasses list
        team_classes = [0] * len(track_classes)
        for cidx, orig in enumerate(indices):
            #Map labels 0->team 1, 1->team 2
            if labels[cidx] == 0:
                team_classes[orig] = 1
            else:
                team_classes[orig] = 2

        #Convert cached colors back to BGR tuples for output
        if self.use_hsv:
            #Need to convert HSV to BGR for display
            outA = cv2.cvtColor(np.uint8([[self.teamA_color]]), cv2.COLOR_HSV2BGR)[0,0]
            outB = cv2.cvtColor(np.uint8([[self.teamB_color]]), cv2.COLOR_HSV2BGR)[0,0]
            outA = tuple(int(c) for c in outA)
            outB = tuple(int(c) for c in outB)
        else:
            outA = tuple(int(c) for c in self.teamA_color)
            outB = tuple(int(c) for c in self.teamB_color)

        #Return the final results
        return {
            "teamAColor": outA,
            "teamBColor": outB,
            "teamClasses": team_classes
        }
