import numpy as np
import cv2

class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier"

    def start(self, data):
        pass

    def stop(self, data):
        pass

    def step(self, data):
        image = data.get("image")
        tracks = data.get("tracks", [])
        track_classes = data.get("trackClasses", [])

        if len(tracks) == 0 or len(track_classes) == 0:
            return {
                "teamAColor": (0, 0, 0),
                "teamBColor": (0, 0, 0),
                "teamClasses": []
            }

        #Spieler extrahieren
        player_colors = []
        player_indices = []
        for idx, (track, cls) in enumerate(zip(tracks, track_classes)):
            if cls == 2:  #Spieler
                x, y, w, h = track.astype(int)
                x1, y1 = max(int(x - w//2), 0), max(int(y - h//2), 0)
                x2, y2 = min(int(x + w//2), image.shape[1]-1), min(int(y + h//2), image.shape[0]-1)
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    mean_color = np.mean(crop.reshape(-1, 3), axis=0)
                    player_colors.append(mean_color)
                    player_indices.append(idx)

        if len(player_colors) < 2:
            team_classes = [0 if c != 2 else 1 for c in track_classes]
            return {
                "teamAColor": (0, 0, 0),
                "teamBColor": (0, 0, 0),
                "teamClasses": team_classes
            }

        #2-Means (numpy)
        player_colors = np.array(player_colors)
        #2 Clusterzentren
        c0, c1 = player_colors[0], player_colors[1]
        for _ in range(10):  
            dists0 = np.linalg.norm(player_colors - c0, axis=1)
            dists1 = np.linalg.norm(player_colors - c1, axis=1)
            labels = (dists1 < dists0).astype(int)
            if np.sum(labels == 0) > 0:
                c0 = np.mean(player_colors[labels == 0], axis=0)
            if np.sum(labels == 1) > 0:
                c1 = np.mean(player_colors[labels == 1], axis=0)
        #Clusterzentren zu Tupel
        teamA_color = tuple(map(int, c0))
        teamB_color = tuple(map(int, c1))

        team_classes = []
        color_idx = 0
        for cls in track_classes:
            if cls == 2:
                team = labels[color_idx]
                team_classes.append(1 if team == 0 else 2)
                color_idx += 1
            else:
                team_classes.append(0)

        return {
            "teamAColor": teamA_color,
            "teamBColor": teamB_color,
            "teamClasses": team_classes
        }
