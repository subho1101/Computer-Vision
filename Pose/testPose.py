import cv2 as cv
import time
import PoseModule as pm


cap = cv.VideoCapture("2.mp4")
p_time = 0
        
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    detector.find_pose(img)    
        
        
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
            
    cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
    cv.imshow("Image", img)
            
    cv.waitKey(10)
    