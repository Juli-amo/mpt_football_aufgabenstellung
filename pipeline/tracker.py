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
            


from scipy.optimize import linear_sum_assignment

class Filter:
    def __init__(self, bbox, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
        # bbox: [x, y, w, h]
        # Initialisiere Zustandsvektor [cx, cy, vx, vy]
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        self.x = np.array([[cx], [cy], [0.], [0.]])
        # Kovarianz
        self.P = np.eye(4) * 1.0
        # Zustandsübergangsmodell
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # Beobachtungsmodell: nur Position
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        # Prozess- und Messrauschen
        self.Q = np.eye(4) * process_noise
        self.R = np.eye(2) * measurement_noise


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

        positions   = np.array([f.position for f in self.tracks], dtype=np.float32)
        velocities  = np.array([f.velocity for f in self.tracks], dtype=np.float32)
        ages        = [f.age for f in self.tracks]
        classes_out = [f.object_class for f in self.tracks]
        ids         = [f.id for f in self.tracks]

        return {
            "tracks":          positions,
            "trackVelocities": velocities,
            "trackAge":        ages,
            "trackClasses":    classes_out,
            "trackIds":        ids
        }





            
        pass




