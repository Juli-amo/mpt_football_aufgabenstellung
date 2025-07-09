import numpy as np
from scipy.optimize import linear_sum_assignment

class Filter:
    def __init__(self, z, cls, track_id):
        self.id = track_id
        self.object_class = cls
        self.age = 1
        self.missing_frames = 0


        self.X = np.array([[z[0]], [z[1]], [0.0], [0.0]])
        self.P = np.diag([10.0, 10.0, 50.0, 50.0]) 

 
        self.R = np.diag([5.0, 5.0])
        # Messmatrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])


        self.Q = np.diag([0.01, 0.01, 0.5, 0.5])

        self.width = z[2]
        self.height = z[3]
        self.last_z = np.array([z[0], z[1]])  

    def predict(self, dt=1.0):
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.X = F @ self.X
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_new, dt=1.0):
        z = np.array([[z_new[0]], [z_new[1]]])

 
        if self.age == 1:
            vx = (z_new[0] - self.last_z[0]) / dt
            vy = (z_new[1] - self.last_z[1]) / dt
            self.X[2, 0], self.X[3, 0] = vx, vy

        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.X = self.X + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.width = z_new[2]
        self.height = z_new[3]

        self.missing_frames = 0
        self.age += 1
        self.last_z = np.array([z_new[0], z_new[1]])

    def no_update(self):
        self.missing_frames += 1
        self.age += 1

    def gating_distance(self, z_new):
        z = np.array([[z_new[0]], [z_new[1]]])
        y = z - self.H @ self.X
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)

    def should_delete(self, max_missing_frames=5):
        return self.missing_frames > max_missing_frames

    @property
    def position(self):
        return np.array([self.X[0, 0], self.X[1, 0], self.width, self.height], dtype=np.float32)

    @property
    def velocity(self):
        return np.array([self.X[2, 0], self.X[3, 0]], dtype=np.float32)

class Tracker:
    def __init__(self):
        self.name = "Tracker"
        self.tracks = []
        self.next_id = 0

    def start(self, data=None):
        self.tracks = []
        self.next_id = 0

    def stop(self, data=None):
        pass

    def step(self, data, dt=1.0):
        detections = data.get("detections", np.empty((0, 4), np.float32))
        det_classes = data.get("classes", np.empty((0,), np.int32))
        N = len(self.tracks)
        M = len(detections)
        gating_threshold = 9.21  
        large_cost = 1e6

        for f in self.tracks:
            f.predict(dt)

        cost = np.full((N, M), large_cost, dtype=np.float32)
        for i, f in enumerate(self.tracks):
            for j, z in enumerate(detections):
                md2 = f.gating_distance(z)
                if md2 <= gating_threshold:
                    cost[i, j] = np.sqrt(md2)

        if N and M:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)

        matched_tracks = set()
        matched_detections = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= large_cost:
                continue
            self.tracks[r].update(detections[c], dt)
            self.tracks[r].object_class = int(det_classes[c])
            matched_tracks.add(r)
            matched_detections.add(c)

        new_tracks = []
        for idx, f in enumerate(self.tracks):
            if idx not in matched_tracks:
                f.no_update()
            if not f.should_delete():
                new_tracks.append(f)
        self.tracks = new_tracks

        for j, z in enumerate(detections):
            if j in matched_detections:
                continue
            f = Filter(z, int(det_classes[j]), self.next_id)
            self.next_id += 1
            self.tracks.append(f)

        positions = np.array([f.position for f in self.tracks], dtype=np.float32)
        velocities = np.array([f.velocity for f in self.tracks], dtype=np.float32)
        ages = [f.age for f in self.tracks]
        classes_out = [f.object_class for f in self.tracks]
        ids = [f.id for f in self.tracks]

        return {
            "tracks": positions,
            "trackVelocities": velocities,
            "trackAge": ages,
            "trackClasses": classes_out,
            "trackIds": ids
        }
