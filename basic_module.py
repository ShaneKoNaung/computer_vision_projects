import cv2
import mediapipe as mp 
import time

class handDetector():
    def __init__(self, mode=False, max_hands = 2, model_complexity = 1, min_det_conf = 0.5, max_tracking_conf = 0.5):
        """
        initialize mediapipe.solutions.hands.Hands()
        """
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_det_conf = min_det_conf
        self.max_tracking_conf = max_tracking_conf

        self.mp_hands = mp.solutions.hands 
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                        self.min_det_conf, self.max_tracking_conf)
        self.mp_draw = mp.solutions.drawing_utils

        self.tip_lm_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """
        params:
            img : Input OpenCV image
            draw : draw the hand landmarks (default: True)
        Returns:
            Return processed image
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        
        return img
    
    def find_positions(self, img, hand_num = 0, draw = True):
        """
        params:
            img : Input OpenCV image
            hand_num : Number of the hand landmark point to draw
            draw : draw a circle at the position of hand_num point 
        Returns:
            Return processed image
        """
        
        hand_lm_list = []

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand_lms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w) , int(lm.y * h)
                    hand_lm_list.append((id, cx, cy))

                    if draw:
                        if id == hand_num:
                            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        self.lm_list = hand_lm_list
        return hand_lm_list
    
    def get_up_fingers(self):
        """
        Returns a list that contains 1s and 0s. 
        1 means finger up. zero, finger down.
        """
        fingers = []
        # Thumb
        if self.lm_list[self.tip_lm_ids[0]][1] > self.lm_list[self.tip_lm_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_lm_ids[id]][2] < self.lm_list[self.tip_lm_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # totalFingers = fingers.count(1)

        return fingers

def main():
    """
    capture video via webcam number 0 and draw connecting hand landmarks on hands
    """
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    pTime = 0
    cTime = 0 


    while True:
        success, img = cap.read()
        if success:
            img = detector.find_hands(img)
            hand_lm_list = detector.find_positions(img)
            
            if hand_lm_list:
                print(hand_lm_list[9])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 5)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()