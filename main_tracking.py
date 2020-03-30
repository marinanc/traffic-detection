import cv2
import numpy as np
from time import sleep
import os

# Minimum rectangle width and height
w_min = 100
h_min = 100

# Error allowed between pixels
offset = 6

# FPS
delay = 60

detec = []
vehicle = 0

# parameters
cap_region_x_begin = 0.8  # start point/total width
cap_region_y_end = 0.4  # start point/total width

# =============== Variable Mouse ================== #
drawing = False
point1 = ()
point2 = ()

drawingTwo = False
pointTwo_1 = ()
pointTwo_2 = ()
Mouse_count = False


# ================================================ #


# =============== Mouse drawing ================== #
def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing
    global pointTwo_1, pointTwo_2, drawingTwo, Mouse_count

    # ----------Mouse 1------- #
    if Mouse_count == False:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing is False:
                drawing = True
                point1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                point2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            Mouse_count = True

    # ----------Mouse 2------- #
    if Mouse_count == True:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawingTwo is False:
                drawingTwo = True
                pointTwo_1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawingTwo is True:
                pointTwo_2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if drawingTwo is True:
                drawingTwo = False
                Mouse_count = False


# ================================================ #

def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture("/home/mcarrizo/cut_olmos_galicia.mp4")
cv2.namedWindow("Video Original")
cv2.setMouseCallback("Video Original", mouse_drawing)

substraction = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if point1 and point2:
        line1 = cv2.line(frame, point1, point2, (255, 127, 0), 3)
    if pointTwo_1 and pointTwo_2:
        rectangle1 = cv2.rectangle(frame, pointTwo_1, pointTwo_2, (255,127,0), 3)
        tempo = float(1 / delay)
        sleep(tempo)

        # Traffic light ROI
        image_np = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        img = image_np[pointTwo_1[1]:pointTwo_2[1], pointTwo_1[0]:pointTwo_2[0]]  # clip the ROI

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

        # Red light
        if cv2.countNonZero(mask) > 0:
            for (i, c) in enumerate(contour):
                (x, y, w, h) = cv2.boundingRect(c)
                validate_contour = (w >= w_min) and (h >= h_min)
                if not validate_contour:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle around the object
                center = find_center(x, y, w, h)
                detec.append(center)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                for (x, y) in detec:
                    if point1 and point2:
                        if (point2[1]+offset) > y > (point1[1]-offset) and point2[0] > x > point1[0]:
                            vehicle += 1
                            success, img_cap = cap.read()
                            # Snapshot
                            if success:
                                cv2.imwrite(os.path.join(os.getcwd(), '%d.png') % vehicle, img_cap)
                            if (point2[1]+offset) > y > (point1[1]-offset) and point2[0] > x > point1[0]:
                                cv2.line(frame, point1, point2, (0, 0, 255), 3)

    cv2.putText(frame, "VEHICULOS: " + str(vehicle), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    # cv2.imshow("Video Original" , cv2.resize(frame, (1000, 600)))
    #cv2.imshow("Detectar", cv2.resize(dilated, (1000, 600)))


    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
