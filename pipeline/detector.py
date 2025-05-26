from ultralytics import YOLO
<<<<<<< Updated upstream
import numpy as np    
=======
import numpy as np
>>>>>>> Stashed changes

class Detector:
    def __init__(self):
        self.name = "Detector" # Do not change the name of the module as otherwise recording replay would break!
<<<<<<< Updated upstream
        self.model = YOLO(r"C:\Users\moham\OneDrive\Dokumente\MPT\mpt_football_aufgabenstellung\modules\yolov8m-football.pt") 

    def start(self, data):
        pass
=======
        self.model = YOLO(r"C:\Users\moham\OneDrive\Dokumente\MPT\mpt_football_aufgabenstellung\modules\yolov8m-football.pt", )
        

    def start(self, data):
        print("YOLO-Klassen:", self.model.names)
>>>>>>> Stashed changes

    def stop(self, data):
        pass

    
    def step(self, data):
        image = data["image"]  # get current image 

        # use YOLO on the image
        results = self.model(image)[0]  #self.model(image) returns a list, we take the first one (current)

<<<<<<< Updated upstream
        bboxes = []
        classes = []

        for box in results.bboxes: #iterate through detected objects 
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract Bounding Box 
            cls = int(box.cls.cpu().numpy())            # assign objekt to a class

            #calculate center(as required), width and hight for each object so that .display can be able to draw a box around the objects 
=======
        boxes = []
        classes = []

        for box in results.boxes: #iterate through detected objects 
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # extract Bounding Box 
            cls = int(box.cls.cpu().numpy())            # assign objekt to a class

            #calculate (as required) center, width and hight for each object so that .display can be able to draw a box around the objects 
>>>>>>> Stashed changes
            x = (x1 + x2) / 2      
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

<<<<<<< Updated upstream
            bboxes.append([x, y, w, h]) #add to the list in a bounding box format (as required)
            classes.append(cls)         #add to object classification the list 

        return { #return in the right format
            "detections": np.array(bboxes, dtype=np.float32),
=======
            boxes.append([x, y, w, h]) #add to the list in a bounding box format (as required)
            classes.append(cls) #add to object classification the list 

        return { #return in the right format
            "detections": np.array(boxes, dtype=np.float32),
>>>>>>> Stashed changes
            "classes": np.array(classes, dtype=np.int32)
         }