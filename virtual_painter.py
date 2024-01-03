import cv2
import numpy as np 
import time
import os
from basic_module import handDetector

brush_thickness = 15
erasure_thickness = 50



folderPath = 'headers'

myList = os.listdir(folderPath)
print(myList)

overlaylist = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}\{imPath}")
    overlaylist.append(image)

print(len(overlaylist))

header = overlaylist[0]
color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
canvas_img = np.zeros((720, 1280, 3), np.uint8)

detector = handDetector(min_det_conf=0.85)

xp, yp = 0, 0
while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. find hand landmarks
    img = detector.find_hands(img)
    lmList = detector.find_positions(img, draw=False)

    if lmList:

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.get_up_fingers()
        # print(fingers)

        # 4. If selection mode - two fingers are up
        if fingers[1] and fingers[2]:
            print("Selection Mode")
            xp, yp = 0, 0 
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), color, cv2.FILLED)

            # checking for click
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlaylist[0]
                    color = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlaylist[1]
                    color = (255, 0, 0)
                elif 800 < x1 < 950:
                    header = overlaylist[2]
                    color = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlaylist[3]
                    color = (0, 0, 0)
        

        # 5. If Drawing mode - index finger is up
        if fingers[1] and fingers[2] == 0:
            print("Drawing Mode")
            cv2.circle(img, (x1, y1), 15, color, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), color, erasure_thickness)
                cv2.line(canvas_img, (xp, yp), (x1, y1), color, erasure_thickness)
            
                
            cv2.line(img, (xp, yp), (x1, y1), color, brush_thickness)
            cv2.line(canvas_img, (xp, yp), (x1, y1), color, brush_thickness)
            
            xp, yp = x1, y1
    
    img_gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
    _, img_inverse = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inverse = cv2.cvtColor(img_inverse, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, img_inverse)
    img = cv2.bitwise_or(img, canvas_img)

    # setting the header image
    img[0:128, 0:1280] = header

    img = cv2.addWeighted(img, 0.5, canvas_img, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas_img)
    cv2.waitKey(1)