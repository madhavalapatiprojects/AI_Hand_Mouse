import cv2
import mediapipe as mp
import time
import numpy as np
import subprocess

class handRecognize():
    def __init__(self, mode=False, maxNumHands=2, detectionConfidence=0.75, trackConfidence=0.75):
        self.mode = mode
        self.maxNumHands = maxNumHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.handDetection = mp.solutions.hands
        self.hands = self.handDetection.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxNumHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackConfidence
        )
        self.handDrawn = mp.solutions.drawing_utils

    def findingHandObjects(self, feed, drawHand=True):
        imgRGB = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)
        self.showresult = self.hands.process(imgRGB)

        if self.showresult.multi_hand_landmarks:
            for lms in self.showresult.multi_hand_landmarks:
                if drawHand:
                    self.handDrawn.draw_landmarks(feed, lms, self.handDetection.HAND_CONNECTIONS)

        return feed

    def handPosition(self, feed, drawHand=True):
        handlist = []
        if self.showresult.multi_hand_landmarks:
            for lms in self.showresult.multi_hand_landmarks:
                for id, landM in enumerate(lms.landmark):
                    h, w, channels = feed.shape
                    channelx, channely = int(w * landM.x), int(h * landM.y)
                    handlist.append((id, channelx, channely))
        return handlist

    def calculateDistance(self, p1, p2, img):
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]

        # Draw line between thumb and index finger
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Calculate midpoint
        midX, midY = (x1 + x2) // 2, (y1 + y2) // 2

        # Calculate Euclidean distance
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Draw Red circle at midpoint
        if distance < 50:
            cv2.circle(img, (midX, midY), 10, (0, 0, 255), cv2.FILLED)
        else:
            # Turn circle Blue when fingers are apart
            cv2.circle(img, (midX, midY), 10,(255, 0, 0), cv2.FILLED)

        return distance


def get_current_volume():
    """
    Returns the current macOS system volume (0–100) as a string.
    Uses osascript to fetch the 'output volume' from volume settings.
    """
    result = subprocess.run(
        ["osascript", "-e", "output volume of (get volume settings)"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()  # e.g. "50"


def set_volume(volume_level):
    """
    Set system volume on macOS using osascript (0–100).
    Args:
        volume_level (float): Desired volume level from 0 to 100.
    """
    # Clamp volume_level to ensure it's between 0 and 100
    volume_level = max(0, min(100, volume_level))

    # Pass that clamped value directly to AppleScript
    subprocess.run(["osascript", "-e", f"set volume output volume {int(volume_level)}"])

    # Print for debugging
    print(f"Set macOS volume: {volume_level}% (osascript scale: {int(volume_level)})")


def main():
    previousT = 0
    cameraCapture = cv2.VideoCapture(0)
    detection = handRecognize()

    while True:
        connection1, feed = cameraCapture.read()
        feed = detection.findingHandObjects(feed)
        handlist = detection.handPosition(feed)

        # Make sure we have at least the thumb (id=4) and index finger (id=8)
        if len(handlist) >= 8:
            thumb = handlist[4]
            indexFinger = handlist[8]

            # Calculate distance between thumb and index finger
            distance = detection.calculateDistance(thumb, indexFinger, feed)

            # Convert that distance to a 0-100 volume percentage
            volume_level = np.interp(distance, [50, 500], [0, 100])

            # Actually set the system volume
            set_volume(volume_level)

            # Get the current system volume and print it
            current_vol = get_current_volume()
            print(f"Distance: {distance:.2f}, Requested Volume: {int(volume_level)}%, "
                  f"Current system volume: {current_vol}")

            # Optionally display volume on the frame
            cv2.putText(feed, f"Volume: {int(volume_level)}%", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # FPS calculation
        CurrentT = time.time()
        fps = 1 / (CurrentT - previousT)
        previousT = CurrentT

        cv2.putText(feed, f"FPS: {int(fps)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Camera Feed", feed)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraCapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
