import time
import HandTrakingModule as htm
import cv2

cap = cv2.VideoCapture(1)
cTime = 0
pTime = 0
detactor = htm.handDetector()

while True:
    success, img = cap.read() 
    img = detactor.findHands(img)
    lmList = detactor.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        print(lmList[4])
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)   

    cv2.imshow("Image", img)
    cv2.waitKey(1)