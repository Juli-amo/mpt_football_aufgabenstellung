class ShirtClassifier:
    def __init__(self):
        self.name = "Shirt Classifier" # Do not change the name of the module as otherwise recording replay would break!
        self.teamA_color = None
        self.teamB_color = None
        self.initialized = False

    def start(self, data):
        self.initialized = False 
        #pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass


    def step(self, data):
        image = data["image"]
        tracks = data["tracks"]
        track_classes = data.get("trackClasses", [])

        
        
        if len(tracks) == 0 or len(track_classes) == 0:
                return {
                  "teamAColor": (0, 0, 0),
                  "teamBColor": (0, 0, 0),
                  "teamClasses": []
              }
                
                
                
                
                
# TODO: Implement processing of a current frame list
        # The task of the shirt classifier module is to identify the two teams based on their shirt color and to assign each player to one of the two teams

        # Note: You can access data["image"] and data["tracks"] to receive the current image as well as the current track list
        # You must return a dictionary with the given fields:
        #       "teamAColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team A
        #       "teamBColor":       A 3-tuple (B, G, R) containing the blue, green and red channel values (between 0 and 255) for team B
        #       "teamClasses"       A list with an integer class for each track according to the following mapping:
        #           0: Team not decided or not a player (e.g. ball, goal keeper, referee)
        #           1: Player belongs to team A
        #           2: Player belongs to team B
        
        return { "teamAColor": None,
                 "teamBColor": None,
                 "teamClasses": None }