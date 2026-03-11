import cv2
import time
import numpy as np
import HandTrakingModule as htm
from pycaw.pycaw import AudioUtilities


#####################################
wCam, hCam = 640, 480
#####################################

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detactor = htm.handDetector(detectionCon=0.7)


device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
# print(f"Audio output: {device.FriendlyName}")
# print(f"- Muted: {bool(volume.GetMute())}")
# print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
# print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]

vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read() 
    img =detactor.findHands(img)
    lmList = detactor.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
        
        length = np.hypot(x2-x1, y2-y1)
        
        vol= np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length < 50:
            cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)
        
        cv2.rectangle(img, (50,150), (85,400), (255,0,0), 3)
        cv2.rectangle(img, (50,int(volBar)), (85,400), (255,0,0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (50,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)