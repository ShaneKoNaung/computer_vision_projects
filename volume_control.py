import cv2
import time
import math
import numpy as np 
from basic_module import handDetector

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# -------------------- parameters ---------------- #

w_cam, h_cam = 640, 480

# ------------------------------------------------ #


cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
pTime = 0

detector = handDetector(min_det_conf=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

vol_range = volume.GetVolumeRange()

min_vol = vol_range[0]
max_vol = vol_range[1]
vol = 0
vol_bar = 400
vol_per = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_positions(img, draw=False)

    if lm_list:

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx, cy = (x1 + x2) // 2 , (y1 + y2) // 2
        
        cv2.circle(img, (x1, y1), 10, (255,0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255,0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
        cv2.circle(img, (cx, cy), 10, (255,0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Hand range 30 - 180
        # Volume Range 0 - 0.38

        vol = np.interp(length, [30, 180], [min_vol, max_vol])
        vol_bar = np.interp(length, [30, 180], [400, 150])
        vol_per = np.interp(length, [30, 180], [0, 100])

        volume.SetMasterVolumeLevel(vol, None)

        if length <= 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f" {str(int(vol_per))} %", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                1, (0, 255, 0), 3)
    
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps : {str(int(fps))}", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 
                1, (255, 0, 0), 3)


    cv2.imshow('Image', img)
    cv2.waitKey(1)