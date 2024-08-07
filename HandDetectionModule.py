import cv2
import mediapipe as mp
import time

class Detector():
    def __int__(self,mode=False,maxhands=2,detectionCon=0.5,trackCon=0.5 ):
       self.mode = mode
       self.maxhands = maxhands
       self.detectionCon = detectionCon
       self.trackCon = trackCon
       self.mpHands = mp.solutions.hands
       self.hands = self.mpHands.Hands( self.mode,self.maxhands,self.detectionCon,self.trackCon )
       self.mpDraw = mp.solutions.drawing_utils






    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        #for fps we need to add this code



def main():
    pTime=0
    cTime=0
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()

        cTime= time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255),3)
        cv2.imshow("Image", img)
        cv2.waitKey(50)





if __name__=="__main__":
    main()
