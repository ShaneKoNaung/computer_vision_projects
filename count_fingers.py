import cv2
import time
from pathlib import Path
from basic_module import handDetector

# -------------- #
wCam, hCam = 640, 480
# -------------- #


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


img_fld = Path("hand_imgs")
img_path_list = list(img_fld.iterdir())


overlay_img_list = []
for img_path in img_path_list:
    image = cv2.imread(str(img_path))
    overlay_img_list.append(image)


pTime = 0
detector = handDetector(min_det_conf=0.75)
tiplm_ids = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_positions(img, draw=False)

    if len(lm_list) != 0:
        fingers = detector.get_up_fingers()

        total_fingers = fingers.count(1)

        h, w, c = overlay_img_list[total_fingers].shape
        img[0:h, 0:w] = overlay_img_list[total_fingers]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)
    
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    

    cv2.imshow("Image", img)
    cv2.waitKey(1)