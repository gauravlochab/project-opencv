from scipy.spatial import distance as dist

import numpy as np
import time
import dlib
import cv2

def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar


COUNTER = 0
TOTAL = 0

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = (48,68)
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(0)
fileStream = False
time.sleep(1.0)

cv2.namedWindow("test")

while True:
    ret, frame = vs.read()
    frame = cv2.resize(frame, (frame.shape[0],450))

    imhsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    imhsvCLAHE = imhsv.copy()

    # Perform histogram equalization only on the V channel
    imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imhsvCLAHE[:,:,2] = clahe.apply(imhsvCLAHE[:,:,2])

    # Convert back to BGR format
    imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)
    imEqCLAHE = cv2.cvtColor(imhsvCLAHE, cv2.COLOR_HSV2BGR)
    frame = imEqCLAHE

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    print("1")
    for rect in rects:
        shape = predictor(gray, rect)
        coords = np.zeros((shape.num_parts, 2), dtype='int')
        print("2")
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        shape = coords
        mouth= shape[mStart:mEnd]
        mar= smile(mouth)
        print("3")
        mouthHull = cv2.convexHull(mouth)
        #print(shape)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
       

        if mar <= .29 or mar > .40 :
            COUNTER += 1
            cv2.putText(frame, "SMILE", (200,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            if COUNTER >= 15:
                TOTAL += 1
                frame = vs.read()
                time.sleep(.3)

            COUNTER = 0
        try:
            cv2.putText(frame, "MAR: {}".format(mar), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            
        except:
            continue
    key2 = cv2.waitKey(1) & 0xFF
    if key2 == ord('q'):
        break
    


vs.release()
cv2.destroyAllWindows()

