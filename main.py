import cv2
import numpy as np
from time import sleep
import os

w_min = 100  # Minimum rectangle width
h_min = 100  # Minimum rectangle height

offset = 6  # Error allowed between pixels

# Lines position
pos_line1 = 435
pos_line2 = 895
pos_line3 = 620

delay = 60  # FPS

detec = []
vehicle = 0

# parameters
cap_region_x_begin = 0.8  # start point/total width
cap_region_y_end = 0.4  # start point/total width


def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('cut3_sjuanycan.mp4')
substraction = cv2.createBackgroundSubtractorKNN(detectShadows=False)  # search history y varTreshold
# substraction = cv2.bgsegm.createBackgroundSubtractorMOG() python2.7

while True:
    ret, frame = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)

    # Traffic light ROI
    image_np = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    img = image_np[285:int(cap_region_y_end * image_np.shape[0]) - 100,  # x
          int(cap_region_x_begin * image_np.shape[1]) - 180:image_np.shape[1] - 520]  # clip the ROI

    cv2.imshow('mask', img)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range of red color in HSV
    lower_red = (170, 100, 100)
    upper_red = (179, 255, 255)
    # Treshold the HSV image to get only red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(frame,frame, mask= mask)

    # Object area
    moments = cv2.moments(mask)
    area = moments['m00']

    if area > 170000:
        # Object center (x,y)
        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])

        # Dibujamos una marca en el centro del objeto
        cv2.rectangle(img, (x, y), (x + 2, y + 2), (0, 0, 255), 2)

    # Showing the mask
    cv2.imshow('mask2', mask)

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = substraction.apply(blur)
    # cv2.imshow('substraction', img_sub)
    dilat = cv2.dilate(img_sub, np.ones((3, 3)))  # (5,5)
    # cv2.imshow('dilate', dilat)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contour, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (750, pos_line1 - 10), (1100, pos_line1), (255, 127, 0), 3)
    cv2.line(frame, (0, pos_line2 - 30), (900, pos_line2), (255, 127, 0), 12)
    cv2.line(frame, (1100, pos_line3 - 30), (1850, pos_line3), (225, 127, 0), 10)

    # Red light
    if cv2.countNonZero(mask) > 0:
        for (i, c) in enumerate(contour):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_contour = (w >= w_min) and (h >= h_min)
            if not validate_contour:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle around the object
            centro = find_center(x, y, w, h)
            detec.append(centro)
            cv2.circle(frame, centro, 4, (0, 0, 255), -1)

            # for (x,y) in detec:
            #     if y<(pos_line1+offset) and y>(pos_line1-offset) and x<1100 and x>700:
            #         vehicle+=1
            #         cv2.line(frame, (750, pos_line1-10), (1100, pos_line1), (0,127,255), 3)
            #         detec.remove((x,y))
            #         print("Vehiculos detectados hasta el momento: "+str(vehicle))

            for (x, y) in detec:
                if ((pos_line1 + offset) > y > (pos_line1 - offset) and 1100 > x > 700) or (
                        (pos_line2 + offset) > y > (pos_line2 - offset) and 900 > x > 0) or (
                        (pos_line3 + offset) > y > (pos_line3 - offset) and 1850 > x > 1100):
                    vehicle += 1
                    success, img_cap = cap.read()
                    # Snapshot
                    if success:
                        cv2.imwrite(os.path.join(os.getcwd(), '%d.png') % vehicle, img_cap)
                    # #Video
                    # out = cv2.VideoWriter(os.path.join(os.getcwd(), '%d.avi') % vehicle,cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, (int(cap.get(3)), int(cap.get(4))))
                    # start_time = time.time()
                    # while( int(time.time() - start_time) < capture_duration ):
                    # 	ret, frame = cap.read()
                    # 	if ret==True:
                    # 		out.write(frame)
                    # 	else:
                    # 		break

                    if (pos_line1 + offset) > y > (pos_line1 - offset) and 1100 > x > 700:
                        cv2.line(frame, (750, pos_line1 - 10), (1100, pos_line1), (0, 0, 255), 3)
                    if (pos_line2 + offset) > y > (pos_line2 - offset) and 900 > x > 0:
                        cv2.line(frame, (0, pos_line2 - 30), (900, pos_line2), (0, 0, 255), 12)
                    if (pos_line3 + offset) > y > (pos_line3 - offset) and 1850 > x > 1100:
                        cv2.line(frame, (1100, pos_line3 - 30), (1850, pos_line3), (0, 0, 255), 10)
                    detec.remove((x, y))
                # print("Cantidad de Vehiculos: "+str(vehicle))

    cv2.putText(frame, "VEHICULOS: " + str(vehicle), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    # cv2.imshow("Video Original", frame)
    cv2.imshow("Video Original", cv2.resize(frame, (1000, 600)))
    cv2.imshow("Detectar", cv2.resize(dilated, (1000, 600)))

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
