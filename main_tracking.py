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


cap = cv2.VideoCapture("/home/mcarrizo/02.mp4")
cv2.namedWindow("Video Original")
cv2.setMouseCallback("Video Original", mouse_drawing)

substraction = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if point1 and point2:
        rectangle1 = cv2.rectangle(frame, point1, point2, (255, 127, 0), 2)
    tempo = float(1 / delay)
    sleep(tempo)

    if point1 and point2:
        img = frame[point1[1]:point2[1], point1[0]:point2[0]]

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = substraction.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((3, 3)))  # (5,5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        contour, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for (i, c) in enumerate(contour):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_contour = (w >= w_min) and (h >= h_min)
            if not validate_contour:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Rectangle around the object
            center = find_center(x, y, w, h)
            detec.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            alpha = 0.4

            for (x, y) in detec:
                if point1 and point2:
                    if (point2[1]+offset) > y > (point1[1]-offset) and point2[0] > x > point1[0]:
                        vehicle += 1
                        overlay = frame.copy()
                        cv2.rectangle(overlay, point1, point2, (0, 0, 255), -1)  # A filled rectangle

                        alpha = 0.4  # Transparency factor.

                        # Following line overlays transparent rectangle over the image
                        image_new = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                        detec.remove((x,y))
                        print("Vehiculo mal estacionado")
                        # Snapshot

    cv2.putText(frame, "VEHICULOS: " + str(vehicle), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    # cv2.imshow("Video Original" , cv2.resize(frame, (1000, 600)))
    #cv2.imshow("Detectar", cv2.resize(dilated, (1000, 600)))

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
