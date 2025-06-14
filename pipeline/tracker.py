import numpy as np

# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track


# VideoReader  →  Detector  →  OpticalFlow  →  Tracker  →  ShirtClassifier  →  Display
# Engine ruft in jedem Frame nacheinander step() aller Module auf.
# Das Detector-Modul läuft vor dem Tracker und schreibt seine Ergebnisse in data:
# data["detections"]  # Nx4  Bounding-Boxen
# data["classes"]     # Nx1  Klassen
# Danach ruft die Engine Tracker.step(data) auf.
# – Der Tracker liest nur data["detections"] und data["classes"].
# – Er muss keinen Detector instanziieren oder aufrufen; das ist schon passiert.
# der Tracker SOLL den Detector NICHT aufrufen. Er nutzt einfach dessen Output
            



class Filter:
    def __init__(self, z, cls, track_id):
        self.id = track_id
        self.position = z # → [x, y, w, h]
        self.last_position = self.position.copy()
        self.object_class = cls # → (ball: 0, player: 2,..)
        self.age = 1            # → since how many frames does the track exist ?
        self.missing_frames = 0 # → since how many frames is the track not detectable ?
        self.velocity = np.array([0.0, 0.0])  # → position change per frame |new postion - old position|
        
<<<<<<< Updated upstream
=======
    
    def update (self, z_new):  #if detection
        self.last_position = self.position
        self.position = z_new
        self.velocity = np.array(self.position[:2]) - np.array(self.last_position[:2])
                        #with numpy it is more effiecient for later calculations but it is exactly the same as: 
                        #[self.position[0] - self.last_position[0], self.position[1] - self.last_position[1]]
        self.missing_frames = 0
        self.age += 1


    def no_update (self):  #if no detection accured
        new_position_predicted = [self.position[0]+self.velocity[0], #x and y change
                                  self.position[1]+self.velocity[1],
                                  self.position[2],                  #width and hight remain the same
                                  self.position[3]
                                  ]
        self.last_position = self.position
        self.position = new_position_predicted
        self.missing_frames += 1
        self.age += 1

    def should_delete(self, max_missing_frames = 5): #returns True when self.missing_frames > max_missing_frames
        if self.missing_frames > max_missing_frames:
            return True
        return False

    
>>>>>>> Stashed changes
    
    def update (self, z_new):  #if detection
        self.last_position = self.position
        self.position = z_new
        self.velocity = np.array(self.position[:2]) - np.array(self.last_position[:2])
                        #with numpy it is more effiecient for later calculations but it is exactly the same as: 
                        #[self.position[0] - self.last_position[0], self.position[1] - self.last_position[1]]
        self.missing_frames = 0
        self.age += 1


    def no_update (self):  #if no detection accured
        new_position_predicted = [self.position[0]+self.velocity[0], #x and y change
                                  self.position[1]+self.velocity[1],
                                  self.position[2],                  #width and hight remain the same
                                  self.position[3]
                                  ]
        self.last_position = self.position
        self.position = new_position_predicted
        self.missing_frames += 1
        self.age += 1

    def should_delete(self, max_missing_frames = 5): #returns True when self.missing_frames > max_missing_frames
        if self.missing_frames > max_missing_frames:
            return True
        return False


class Tracker:
    pass

