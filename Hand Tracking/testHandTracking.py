import cv2
import time
import handTrackingModule as htm

p_time = 0

cap = cv2.VideoCapture(0)
detector = htm.hand_detector()
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lm_list = detector.findPosition(img, draw=False)
    if len(lm_list) != 0:
        print(lm_list[4])

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Exit the program on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
