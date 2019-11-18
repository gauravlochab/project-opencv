
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	
	C = dist.euclidean(eye[0], eye[3])

	
	ear = (A + B) / (2.0 * C)

	
	return ear
 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 

EYE_AR_THRESH = 0.23
EYE_AR_THRESH_dr = 0.25
EYE_AR_CONSEC_FRAMES = 3
EYE_AR_CONSEC_FRAMES_dr = 48


COUNTER = 0
ALARM_ON = False
COUNTER_dr = 0
TOTAL = 0



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



vs = VideoStream(src=0).start()

fileStream = False
time.sleep(1.0)


while True:
	
	if fileStream and not vs.more():
		break

	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	
	rects = detector(gray, 0)

	
	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		
		ear = (leftEAR + rightEAR) / 2.0

		
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

		
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		
		else:
			
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			
			COUNTER = 0
		if ear < EYE_AR_THRESH_dr:
			COUNTER_dr += 1
 
			
			if COUNTER_dr >= EYE_AR_CONSEC_FRAMES_dr:
				
				if not ALARM_ON:
					print("ALRAM ON")
					ALARM_ON  = True
                    
 
					
				cv2.putText(frame, "DROWSINESS ALERT!", (100, 300),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 
		
		else:
			COUNTER_dr = 0
			ALARM_ON = False

		
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (100,30 ),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		
 
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
