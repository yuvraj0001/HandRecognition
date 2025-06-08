import cv2
import mediapipe as mp
import time


class HandDetectior:
    def __init__(self,  mode=False, maxHands=2, detectionConfiedence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfiedence= detectionConfiedence
        self.trackingConfidence=trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfiedence,
            min_tracking_confidence=self.trackingConfidence
        )        
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img



def main():

    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetectior()

    while True:
        success, img = cap.read()
        if not success:
            continue  # Skip the rest of the loop
        img = detector.findHands(img)

        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        # print(fps)

        cv2.putText(img, f'FPS : {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

     
if __name__=="__main__":
    main()