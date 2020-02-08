import cv2, sys
import numpy as np



cap = cv2.VideoCapture(3)

if cap is None:
    sys.exit(1)

while(1):
    ret, frame = cap.read()
    if frame is None:
        continue
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       
    lower = np.array([30,100,60])
    upper = np.array([80,255,255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    dst = cv2.cornerHarris(mask, 2, 3, 0.04)
    
    cv2.imshow("Input", frame)
    cv2.imshow("Output", dst)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
sys.exit(0)