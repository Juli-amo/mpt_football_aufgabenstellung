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

        def __init__(self):
            self.name = "Tracker"
            self.tracks = []
            self.next_id = 0

        def start(self, data):
            self.tracks = []
            self.next_id = 0

        def stop(self, data):
            pass

        def _distance(self, p, q):
            return np.linalg.norm(np.array(p) - np.array(q))
        
        def step(self, data):
            detections = data.get("detections", np.empty((0, 4), np.float32))
            det_classes = data.get("classes", np.empty((0,), np.int32))
            N = len(self.tracks)
            M = len(detections)
            MAX_DIST = 300.0

            if N and M:
                cost = np.zeros((N, M), dtype=np.float32)
                for i, f in enumerate(self.tracks):
                    p = f.position[:2]
                    for j, z in enumerate(detections):
                        q = z[:2]
                        dist = self._distance(p, q)
                        cost[i, j] = dist if dist < MAX_DIST else 1e6
                row_ind, col_ind = linear_sum_assignment(cost)
            else:
                cost = np.empty((0, 0), dtype=np.float32)
                row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

            matched_tracks = set()
            matched_detections = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e6:
                    continue
            self.tracks[r].update(detections[c])
            matched_tracks.add(r)
            matched_detections.add(c)
            self.tracks[r].object_class = int(det_classes[c])

        new_track_list = []
        for idx, f in enumerate(self.tracks):
            if idx not in matched_tracks:
                f.no_update()
            if not f.should_delete():
                new_track_list.append(f)
        self.tracks = new_track_list

        for j, z in enumerate(detections):
            if j in matched_detections:
                continue
            f = Filter(z, int(det_classes[j]), self.next_id)
            self.next_id += 1
            self.tracks.append(f)



            
            pass




