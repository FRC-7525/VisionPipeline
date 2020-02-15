import cv2, sys
import numpy as np
from collections import defaultdict


def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A,b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

def segmentedIntersection(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            intersections.append(intersection(lines[i], lines[j]))
    return intersections

def splitByAngle(lines):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    attempts = 10
    
    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)] for angle in angles], dtype = np.float32)
    
    labels, centers = cv2.kmeans(pts, 3, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)
    
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    
    avgLines = []
    
    for i in range(len(segmented)):
        reshaped = np.reshape(segmented[i], (-1, 2))
        average = np.average(reshaped, axis=0)
        avgLines.append(average)
    
    return avgLines

def drawLines(lines,frame):
    for line in lines:
        
        subRho = line[0]
        subTheta = line[1]
        
        a = np.cos(subTheta)
        b = np.sin(subTheta)
        
        x0 = a * subRho
        y0 = b * subRho
        
        x1 = int(np.rint(x0 + 1000 * (-1 * b)))
        y1 = int(np.rint(y0 + 1000 * (a)))
        x2 = int(np.rint(x0 - 1000 * (-1 * b)))
        y2 = int(np.rint(y0 - 1000 * (a)))
        
        cv2.line(frame, (x1,y1), (x2, y2), (0,0,255), 1)

def drawIntersections(intersections, frame):
    for x, y in intersections:
        try:
            cv2.circle(frame, (x,y), 3, (255,0,0), -1)
        except Exception as ex:
            print("Unknown Error: " + str(ex))


cap = cv2.VideoCapture("/dev/video0")

       
lower = np.array([30,100,60])
upper = np.array([80,255,255])


cannyThreshLower = 100
cannyThreshUpper = 200

rho = 1
theta = np.pi/180
houghThreshold = 75

if cap is None:
    sys.exit(1)

while(1):
    ret, frame = cap.read()
    if frame is None:
        continue
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower, upper)
    
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    edges = cv2.Canny(mask, cannyThreshLower, cannyThreshUpper)
    
    lines = cv2.HoughLines(edges, rho, theta, houghThreshold)
    
    if lines is None:
        continue
    
    segments = []
    intersections = []
    
    try:
        segments = splitByAngle(lines)
        flattened = [[line[0], np.pi/2] for line in segments]
        #intersections = segmentedIntersection(flattened)
    except Exception as ex:
        print("Linear Algebra Error: " + str(ex))
    
    #if len(intersections) == 0:
        #continue
    
    drawLines(segments,frame)
    
    #drawIntersections(intersections, frame)
                                                       
    cv2.imshow("Intersections", frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
sys.exit(0)


    