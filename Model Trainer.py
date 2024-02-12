import cv2
import numpy as np
from PIL import Image #pillow package
import os
path = 'C://Users//Prajyot//OneDrive//Desktop//New//Face//samples' 

recognizer = cv2.face.LBPHFaceRecognizer_create() # Local Binary Patterns Histograms
detector = cv2.CascadeClassifier("C://Users//Prajyot//AppData//Local//Programs//Python//Python311//Lib//site-packages//cv2//data//haarcascade_frontalface_default.xml")



def Images_And_Labels(path): 
    paths = 'C://Users//Prajyot//OneDrive//Desktop//New//Face//samples'
    imagePaths = [os.path.join(path,f) for f in os.listdir(paths)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths: 

        gray_img = Image.open(imagePath).convert('L') 
        img_arr = np.array(gray_img,'uint8') 

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_arr)

        for (x,y,w,h) in faces:
            faceSamples.append(img_arr[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("Training faces. It will take a few seconds. Wait ...")

faces,ids = Images_And_Labels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('C://Users//Prajyot//OneDrive//Desktop//New//Face//trainer//trainer.yml')  

print("Model trained, Now we can recognize your face.")
