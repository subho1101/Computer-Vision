import cv2 as cv
import mediapipe as mp
import time


class PoseDetector():
    
    def __init__(self, mode=False, upper_body=False, smooth=True, detection_con=0.5, track_con=0.5):
        
        self.mode = mode
        self.upper_body = upper_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode, smooth_landmarks=self.smooth, min_detection_confidence=self.detection_con, min_tracking_confidence=self.track_con)


        
    def findPose(self, img, draw=True):
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,       self.mp_pose.POSE_CONNECTIONS)

        return img
    
    def getPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmakrs:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
                    
        return lmList
        
       
def main():
    cap = cv.VideoCapture("Pose/2.mp4")
    p_time = 0
    
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)    
        lmList = detector.getPosition(img)
        print(lmList)

        c_time = time.time()
        fps = 1/(c_time - p_time)
        p_time = c_time
        
        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)
        
        cv.waitKey(10)
        
if __name__ == "__main__":
    main()

    
    
def main():
    cap = cv.VideoCapture("Pose/2.mp4")
    p_time = 0
    
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        detector.find_pose(img)    
    
    
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        
        cv.putText(img, str(int(fps)), (70,50), cv.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
        cv.imshow("Image", img)
        
        cv.waitKey(10)
    
    
    
if __name__ == "__main__":
    main()
