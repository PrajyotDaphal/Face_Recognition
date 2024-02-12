from os import environ
environ ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import cv2
from features.Offline.wake import wake
from Extention.say import say
from features.System.Wishme import wish
import pygame
import random

pygame.mixer.init()
startup = pygame.mixer.Sound("C://Users//Prajyot//OneDrive//Desktop//New//wav//binary_coded_computer.mp3")

startup.play()
say("Checking Network Connection...")
print(" ")
say("Connecting To Network Connection...")
print(" ")
say("Connecting jarvis...")
print(" ")
random_strings = ["17B5KB916MB" , "14JH5HDRF3" , "EFS8GGFADD56" , "WE62ETRR199" , "952VCXVGX" , "ZXS5346JFHTT" , "VV2RRW4554" , "WD45JTG537"]
random_string = random.choice(random_strings)
print("Host ID - " + "\033[93m" +  random_string + "\033[0m"+"\n")
say("------------------------------------------------------")
say("Look At The Camera")
say("Verifying")

def face_recognition():

 recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
 recognizer.read('C://Users//Prajyot//OneDrive//Desktop//New//Face//trainer/trainer.yml')   
 cascadePath = "C://Users//Prajyot//AppData//Local//Programs//Python//Python311//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml"
 faceCascade = cv2.CascadeClassifier(cascadePath) 

 font = cv2.FONT_HERSHEY_SIMPLEX 


 id = 1 


 names = ['','Prajyot','roy']  


 cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) #cv2.CAP_DSHOW to remove warning
 cam.set(3, 640) # set video FrameWidht
 cam.set(4, 480) # set video FrameHeight

 
 minW = 0.1*cam.get(3)
 minH = 0.1*cam.get(4)

 ret, img =cam.read() 

 converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  

 faces = faceCascade.detectMultiScale( 
        converted_image,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

 for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) 

        id, accuracy = recognizer.predict(converted_image[y:y+h,x:x+w]) 

        if (accuracy < 100):
            id = names[id]
            accuracy = "  {0}%".format(round(100 - accuracy))
            say("Identity Verify!!!")
            wish()
            cam.release()
            wake()
            break

        else:
            say("Unkonwn ID")
            id = "Unknown ID"
            accuracy = "  {0}%".format(round(100 - accuracy))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(accuracy), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
 cv2.imshow('camera',img) 

 cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video


 # Do a bit of cleanup
 say("Unknown ID")
 cam.release()
 cv2.destroyAllWindows()
 return id
