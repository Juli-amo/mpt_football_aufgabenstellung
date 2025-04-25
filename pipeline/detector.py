from ultralytics import YOLO
    

class Detector:
    def __init__(self):
        self.name = "Detector" # Do not change the name of the module as otherwise recording replay would break!
        self.model = YOLO(r"C:\Users\moham\OneDrive\Dokumente\MPT\mpt_football_aufgabenstellung\modules\yolov8m-football.pt") 

    def start(self, data):
        pass

    def stop(self, data):
        pass

