import cv2
import mediapipe as mp


class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        # Initializes a new Hands instance with default Constructor Arguments
        self.hands = self.mpHands.Hands()
        # Getting an Instance of Drawing_Utils and assigning it to mpDraw
        self.mpDraw = mp.solutions.drawing_utils
        # Represents the Tips of the Fingers
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Returns an array of NamedTuples of Coordinates representing Hand Positions namely "multi_hand_landmarks"
        results = self.hands.process(imgRGB)

        # Checking if the Landmarks exists
        if results.multi_hand_landmarks:
            # Iterate through the Landmarks
            for handLms in results.multi_hand_landmarks:
                # Draw defaults to True
                if draw:
                    # Draws HAND_CONNNECTIONS on top of Images according to the Landmarks
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)
        # Returns the Modified Image
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []

        # Converts the Image Color Scheme from RGB to BGR
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processes the Image and returns an list of NamedTuples -> Hand Landmarks
        results = self.hands.process(imgRGB)

        # Checking if the Landmarks exists
        if results.multi_hand_landmarks:
            # Iterate through the Landmarks
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # Get the Height and Width for the Image
                    h, w = img.shape
                    # Scales the Landmarks according to the Height and Width of the Image
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # Appends the Modified Landmarks to the List
                    self.lmList.append([id, cx, cy])
                    # Draw defaults to True but might be False according to the Conditions
                    if draw:
                        # Draws HAND_CONNNECTIONS on top of Images according to the Landmarks
                        self.mpDraw.draw_landmarks(
                            img, handLms, self.mpHands.HAND_CONNECTIONS)
                        # Draws circles at the Tips of the Hands
                        cv2.circle(img, (cx, cy), 15,
                                   (255, 0, 255), cv2.FILLED)
        # Returns the Modified Landmark List
        return self.lmList

    def fingersUp(self, lmList):
        fingers = []

        # Totally Shitty Method to do this but Yeah !!
        if len(lmList) != 0:
            if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):
                if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        else:
            # Defaulting to [0, 0, 0, 0, 0] in case the Landmarks are Absent
            fingers = [0, 0, 0, 0, 0]
        return fingers
