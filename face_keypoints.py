

import numpy as np
import cv2
import time
import requests
import pandas as pd
from tensorflow.keras.models import load_model





model = load_model('Pre-Trained-models/face_keypoint_detection_v2.model')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')

print('model load complete')


def_x,def_y =  640,480

cap = cv2.VideoCapture(0)

start = time.time()

cv2.destroyAllWindows()
while True:
    
    ret,ori_img = cap.read() 
    #print(ori_img.shape)
    start = time.time()
    img = cv2.resize(ori_img,(200,200))
    #img = cv2.imread('testface.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        x,w =  int((x/200)*def_x),int((w/200)*def_x)
        y,h =  int((y/200)*def_y),int((h/200)*def_y)+20
        face_o = ori_img[y:y+h,x:x+w]
        temp_x,temp_y, _ = face_o.shape
        face_o = cv2.resize(face_o,(120,120))
        face = cv2.cvtColor(face_o, cv2.COLOR_BGR2GRAY)
        face = cv2.Canny(face,50,100)/255

        
        pred = model.predict(face.reshape(1,120,120,1))[0]
        pred[::2] = (pred[::2]/120)*temp_y
        pred[1::2]= (pred[1::2]/120)*temp_x 
        pred = pred.astype('int')
        
        
        color = (135,206,235)

        cv2.circle(ori_img,(x+pred[0],y+pred[1]-10), 2, color, 2)
        cv2.circle(ori_img,(x+pred[2],y+pred[3]-10 ), 2, color, 2)

        cv2.circle(ori_img,(x+pred[4],y+pred[5]-10), 2, color, 2)

        cv2.circle(ori_img,(x+pred[6],y+pred[7]-10 ), 2, color, 2)
        cv2.circle(ori_img,(x+pred[8],y+pred[9]-10 ), 2, color, 2)
        
       
    try:
    	print('fps:', 1/(time.time()-start))
    except:pass
   
    cv2.imshow('face',ori_img)
   
    if cv2.waitKey(1)==27:
        cv2.destroyAllWindows()
        break
        




