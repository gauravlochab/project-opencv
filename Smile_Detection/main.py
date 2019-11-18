
from imutils.video import VideoStream
from imutils import face_utils

import dlib,time,cv2,os

shape_predictor="shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
print("[INFO] STARTING")

vs = VideoStream(src=0).start()

time.sleep(2.0)
j, dist_smile_val,  diff_chx,  diff_chy, cnt =0,0, 0,0, 0

while True:
        
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        diff_smile=0
      
        cv2.imshow("Frame", frame)
        
        x49,y49,x55,y55=0,0,0,0
        dist_smile=0
        for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                i=1
                x1,y1,w,h=0,0,0,0
                j=j+1
                
                for (x, y) in shape:
                        
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                        

                        
                        if(i==49):
                                x49=x
                                y49=y
                        elif(i==55):
                                x55=x
                                y55=y
                                dist_smile=((x49-x55)**2+(y49-y55)**2)**0.5
                                
                                if j==1:
                                        dist_smile_val = dist_smile
                                
                                if dist_smile_val>dist_smile:
                                        dist_smile_val = dist_smile
                                
                                if (dist_smile-dist_smile_val)>20:
                                        dist_smile_val = dist_smile
                                if abs(dist_smile-dist_smile_val)<1 and j!=1:
                                        dist_smile_val=dist_smile_val-5


         
                        if diff_chx<10 and diff_chy<10:
                               
                                
                                if dist_smile-dist_smile_val>10  and j!=1:
                                        cnt = cnt +1 
                                        
                                        if cnt>10:
                                                cv2.putText(frame,'Smile', (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
                                                s=1

                                        if cnt>14:
                                                dist_smile_val=dist_smile
                                else:
                                        cnt=0
                      

                        i=i+1


       
        cv2.imshow("Frame", frame)
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
 
VideoStream(src=0).stop()
cv2.destroyAllWindows()