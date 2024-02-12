import cv2

def face_recognition():

 recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
 recognizer.read('Your trainer.yml File Path')   
 cascadePath = "your haarcascade_frontalface_default.xml File Path"
 faceCascade = cv2.CascadeClassifier(cascadePath) 

 font = cv2.FONT_HERSHEY_SIMPLEX 

 id = 1 

 names = ['','roy']  

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
            cam.release()

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
