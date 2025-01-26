import cv2
import mediapipe as mp
import time

cameraCapture = cv2.VideoCapture(0)

handDetection = mp.solutions.hands
hands = handDetection.Hands()
handDrawn = mp.solutions.drawing_utils


previousT = 0
CurrentT = 0

while True:
    connection1, feed = cameraCapture.read()
    imgRGB = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)
    showresult = hands.process(imgRGB)

    if showresult.multi_hand_landmarks:
        for lms in showresult.multi_hand_landmarks:
            for id, landM in enumerate(lms.landmark):
                h, w, channels = feed.shape
                channelx, channely = int(w*landM.x), int(h*landM.y)
                print(id, channelx, channely)
                if (id == 4):
                    cv2.circle(feed, (channelx, channely), 17, (255,0,0), cv2.FILLED)

            handDrawn.draw_landmarks(feed, lms, handDetection.HAND_CONNECTIONS)

    CurrentT = time.time()
    fps = 1/(CurrentT-previousT)
    previousT = CurrentT

    cv2.putText(feed, str(int(fps)), (30,100), cv2.FONT_ITALIC, 4, (255,255,0), 4)

    cv2.imshow("Camera Feed", feed)
    cv2.waitKey(1)