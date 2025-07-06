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

    def predict(self):
        # Vorhersagezustand
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, bbox):
        # Update mit neuer Messung
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        z = np.array([[cx], [cy]])
        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_residual
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        # Gibt aktuellen Mittelpunkt und Geschwindigkeit zurück
        return self.x.flatten()  # [cx, cy, vx, vy]

class Tracker:
    def __init__(self, max_skipped=5, max_trace_length=10, dist_thresh=50):
        self.next_track_id = 0
        self.tracks = {}  # tid: {'filter', 'trace', 'skipped', 'age', 'prev_pos', 'class'}
        self.max_skipped = max_skipped
        self.max_trace_length = max_trace_length
        self.dist_thresh = dist_thresh

    def step(self, data):
        detections = data.get("detections", [])
        det_classes = data.get("classes", []) 

        # das ist die Vorhersage
        for tid, tr in list(self.tracks.items()):
            tr['filter'].predict()
            tr['skipped'] += 1
            tr['age'] += 1

        # Hier passiert die Zuordnung
        assignments = []
        unassigned = []
        for i, det in enumerate(detections):
            best_tid, best_dist = None, float('inf')
            x, y, w, h = det
            cx_det = x + w / 2
            cy_det = y + h / 2
            for tid, tr in self.tracks.items():
                cx_pred, cy_pred, _, _ = tr['filter'].get_state()
                dist = np.hypot(cx_det - cx_pred, cy_det - cy_pred)
                if dist < best_dist:
                    best_dist, best_tid = dist, tid
            if best_dist < self.dist_thresh and best_tid is not None:
                assignments.append((best_tid, i, det))
            else:
                unassigned.append((i, det))

        # hier werden die zugeorndeten tracks geupdated
        matched_ids = set()
        for tid, i, det in assignments:
            tr = self.tracks[tid]
            prev = tr['filter'].get_state()[:2]
            tr['filter'].update(det)
            cx, cy, _, _ = tr['filter'].get_state()
            tr['trace'].append((cx, cy))
            if len(tr['trace']) > self.max_trace_length:
                tr['trace'].pop(0)
            tr['skipped'] = 0
            tr['prev_pos'] = prev
            tr['object_class'] = int(det_classes[i]) if i < len(det_classes) else -1
            matched_ids.add(tid)

        # neue tracks für nicht zugeordnete
        for i, det in unassigned:
            filt = Filter(det)
            cx, cy, _, _ = filt.get_state()
            self.tracks[self.next_track_id] = {
                'filter': filt,
                'trace': [(cx, cy)],
                'skipped': 0,
                'age': 1,
                'prev_pos': (cx, cy),
                'object_class': int(det_classes[i]) if i < len(det_classes) else -1
            }
            self.next_track_id += 1

        # vorheriger zustand entfernt 
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['skipped'] > self.max_skipped:
                del self.tracks[tid]

        positions = []
        velocities = []
        ages = []
        classes_out = []
        ids = []

        for tid, tr in self.tracks.items():
            cx, cy, _, _ = tr['filter'].get_state()
            px, py = tr['prev_pos']
            vx = cx - px
            vy = cy - py
            positions.append([cx, cy])
            velocities.append([vx, vy])
            ages.append(tr['age'])
            classes_out.append(tr['object_class'])
            ids.append(tid)

        return {
            "tracks":          np.array(positions, dtype=np.float32),
            "trackVelocities": np.array(velocities, dtype=np.float32),
            "trackAge":        ages,
            "trackClasses":    classes_out,
            "trackIds":        ids
        }





            
        pass




