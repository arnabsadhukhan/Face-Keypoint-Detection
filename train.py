import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Dense,Flatten,Input,Conv2D
from tensorflow.keras.callbacks import ModelCheckpoint

print('import complete')

datadir = 'C:/Users/arnab/Python/Celeba/Face Keypoint Detection/celeba-dataset/' # change the directory


#NOW LOAD THE LANDMARKS CSV FILE

face_keypoints = pd.read_csv(datadir + 'list_landmarks_align_celeba.csv')

face_keypoints = face_keypoints.set_index('image_id')
print('preview of csv file:')
print(face_keypoints.head())

datadir_face_img = datadir + 'img_align_celeba/img_align_celeba/'

"""LETS PLOT THIS ATTRIBUTES"""

for image_name in os.listdir(datadir_face_img)[:5]:
  img  =  cv2.imread(datadir_face_img+image_name,0)
  
  (x,y,w,h) = 30,60,120,120
  img = img[y:y+h,x:x+w]
  img = cv2.Canny(img,50,150)
  

  attributes = face_keypoints.loc[image_name]
  color = (135,206,235)

  cv2.circle(img,(attributes.lefteye_x-x,attributes.lefteye_y-y ), 2, color, 2)
  cv2.circle(img,(attributes.righteye_x-x,attributes.righteye_y-y ), 2, color, 2)

  cv2.circle(img,(attributes.nose_x-x,attributes.nose_y-y ), 2, color, 2)

  cv2.circle(img,(attributes.leftmouth_x-x,attributes.leftmouth_y-y ), 2, color, 2)
  cv2.circle(img,(attributes.rightmouth_x-x,attributes.rightmouth_y-y ), 2, color, 2)

  plt.figure()
  plt.imshow(img)

"""CROP THE IMAGES AND ALL ADJUST THE X ,Y COORDINATES ACCORDING TO CROP IMAGE"""

(x,y,w,h) = 30,60,120,120

face_keypoints.loc[:,'lefteye_x'] = (face_keypoints.loc[:,'lefteye_x'] -x)
face_keypoints.loc[:,'righteye_x'] = (face_keypoints.loc[:,'righteye_x'] -x)
face_keypoints.loc[:,'nose_x'] = (face_keypoints.loc[:,'nose_x'] -x)
face_keypoints.loc[:,'leftmouth_x'] = (face_keypoints.loc[:,'leftmouth_x'] -x)
face_keypoints.loc[:,'rightmouth_x'] = (face_keypoints.loc[:,'rightmouth_x'] -x)

face_keypoints.loc[:,'lefteye_y'] = (face_keypoints.loc[:,'lefteye_y'] -y)
face_keypoints.loc[:,'righteye_y'] = (face_keypoints.loc[:,'righteye_y'] -y)
face_keypoints.loc[:,'nose_y'] = (face_keypoints.loc[:,'nose_y'] -y)
face_keypoints.loc[:,'leftmouth_y'] = (face_keypoints.loc[:,'leftmouth_y'] -y)
face_keypoints.loc[:,'rightmouth_y'] = (face_keypoints.loc[:,'rightmouth_y'] -y)

print('preview after adjust the COORDINATES:')
print(face_keypoints.head())


print('check the new COORDINATES')
for image_name in os.listdir(datadir_face_img)[:5]:
  img  =  cv2.imread(datadir_face_img+image_name,0)
  (x,y,w,h) = 30,60,120,120
  img = img[y:y+h,x:x+w]
  attributes = face_keypoints.loc[image_name]
  color = (135,206,235)

  cv2.circle(img,(int(attributes.lefteye_x),int(attributes.lefteye_y )), 2, color, 2)
  cv2.circle(img,(int(attributes.righteye_x),int(attributes.righteye_y) ), 2, color, 2)

  cv2.circle(img,(int(attributes.nose_x),int(attributes.nose_y) ), 2, color, 2)

  cv2.circle(img,(int(attributes.leftmouth_x),int(attributes.leftmouth_y) ), 2, color, 2)
  cv2.circle(img,(int(attributes.rightmouth_x),int(attributes.rightmouth_y) ), 2, color, 2)
  plt.figure()
  plt.imshow(img)

"""CREATE TRAINING DATA"""
X = []  # all images 
points = []  # all attributes
i = 0
batch = 1000
print('CREATE TRAINING DATA')
for image_name in tqdm(os.listdir(datadir_face_img)[batch*i:(i+1)*batch]):
  img  =  cv2.imread(datadir_face_img+image_name,0) # read the image in gray scale
  (x,y,w,h) = 30,60,120,120   # crop the images with this 
  img = img[y:y+h,x:x+w]  # crop the images 
  img = cv2.Canny(img,50,150)   # apply the edge detection 
  attributes = face_keypoints.loc[image_name]
  img = img/255.0
  X.append(img)
  points.append(list(attributes))


X = np.array(X).reshape(-1,120,120,1)
points = np.array(points).reshape(-1,10)


"""NOW CREATE THE MODEL"""

input_layer = Input(shape= ( 120,120,1))

block1 = Conv2D(16,(3,3),padding='same',activation='relu')(input_layer)
block2 = Conv2D(8,(3,3),padding='same',activation='relu')(block1)

flat = Flatten()(block2)

flat = Dense(50,activation='relu')(flat)
output_layer = Dense(10)(flat)


model = Model(inputs=input_layer,outputs=output_layer)
model.compile(optimizer='adam', loss='mse',metrics = ['accuracy'])

print('model build complete:')
print(model.summary())





model_name = 'face_keypoint_detection_v3.model'
callback = ModelCheckpoint(filepath = model_name ,monitor = 'val_accuracy',save_best_only=True,mode = 'max')

acc = []
val_acc=[]
loss = []
val_loss = []

print('TRAINING START')
model.fit(X,points,epochs=1,validation_split=0.3,batch_size=32,callbacks=[callback])

model.save('final'+'-'+model_name)

print('Finish')