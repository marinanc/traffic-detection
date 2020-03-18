import cv2
import numpy as np
import time
from time import sleep
import os

#Video capture
capture_duration=1
delay= 60 #FPS

#Rectangle
w_min=100
h_min=100

#Error allow between pixels
offset=6

#Lines position
pos_line1=435 
pos_line2=895
pos_line3=620

#FPS
delay=60

#Vehicles
detec=[]
vehicle=0

# parameters
cap_region_x_begin=0.8  # start point/total width
cap_region_y_end=0.4  # start point/total width

	
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

def start_detection(x,y,detect,vehicle):
	for (x,y) in detect:
		vehicle+=1
		succes,img_cap = cap.read()

		if succes:
			out = cv2.VideoWriter(os.path.join(os.getcwd(), '%d.avi') % vehicle,cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, (int(cap.get(3)), int(cap.get(4))))
			start_time = time.time()
			while(int(time.time() - start_time) < capture_duration):
				ret, frame = cap.read()
				if ret==True:
					out.write(frame)
				else:
					break

		if y<(pos_line1+offset) and y>(pos_line1-offset) and x<1100 and x>700:
			cv2.line(frame1, (750, pos_line1-10), (1100, pos_line1), (0,0,255), 3)
		elif y<(pos_line2+offset) and y>(pos_line2-offset) and x<900 and x>0:
			cv2.line(frame1, (0, pos_line2-30), (900, pos_line2), (0,0,255), 12)
		elif y<(pos_line3+offset) and y>(pos_line3-offset) and x<1850 and x>1100:
			cv2.line(frame1, (1100, pos_line3-30), (1850, pos_line3), (0,0,255), 10)
		detect.remove((x,y))

		return vehicle

cap = cv2.VideoCapture('sjuanycan_4hs.mp4')
substraction = cv2.createBackgroundSubtractorKNN(detectShadows=False)

while True:
    ret , frame = cap.read()
    tempo = float(1/delay)
    sleep(tempo)

    #Traffic light ROI 
    image_np = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    img = image_np[285:int(cap_region_y_end * image_np.shape[0])-100, #x
                int(cap_region_x_begin * image_np.shape[1])-180:image_np.shape[1]-520]  # clip the ROI
    cv2.imshow('mask', img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Range of red color in HSV
    lower_red = (170,100,100)
    upper_red = (179,255,255)
    #Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)


    #Area of objects
    moments = cv2.moments(mask)
    area = moments['m00']
    if(area > 170000): 
        #Center x,y of the object
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])
        #Drawing a mark in the center
        cv2.rectangle(img, (x, y), (x+2, y+2),(0,0,255), 2)
     
    #Showing the mask
    cv2.imshow('mask2', mask)

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = substraction.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilat, cv2.MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2.MORPH_CLOSE , kernel)
    
    contour,h = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (750, pos_line1-10), (1100, pos_line1), (255,127,0), 3) 
    cv2.line(frame, (0, pos_line2-30), (900, pos_line2), (255,127,0), 12)
    cv2.line(frame, (1100, pos_line3-30), (1850, pos_line3), (225,127,0), 10)

    #Red light
    if cv2.countNonZero(mask) > 0:
	    for(i,c) in enumerate(contour):
	        (x,y,w,h) = cv2.boundingRect(c)
	        validate_contour = (w >= w_min) and (h >= h_min)
	        if not validate_contour:
	            continue

	        center = find_center(x, y, w, h)
	        detec.append(center)
	        cv2.circle(frame, center, 4, (0, 0,255), -1)

	        vehicle=start_detection(x,y,detec,vehicle)

    cv2.imshow("Traffic" , cv2.resize(frame, (1000, 600)))
    cv2.imshow("Detecting", cv2.resize(dilated, (1000, 600)))

    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
