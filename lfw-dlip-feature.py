from imutils import face_utils
import dlib
import cv2
import numpy as np
from numpy import save
import os
# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face
path_dataset="C:/Users/pnc/Documents/impact.ai/mask-feature-exp/lfw-dataset/test/"
path_to_read="C:/Users/pnc/Documents/impact.ai/mask-feature-exp/lfw-deepfunneled/"
directory = os.path.join(path_to_read)
# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

for root,dirs,files in os.walk(directory):
    
    for direc in dirs:
        fil=os.path.join(directory + direc)
        path = os.path.join(path_dataset,direc)
        print(path)
        os.mkdir(path)
        for toots,dirss,filess in os.walk(fil):
                
                for i in range(len(filess)):
                    
                    if files==None:
                        print(filess)
                    image = cv2.imread(fil + "/" + filess[i])
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray, 0)
                    for (x, rect) in enumerate(rects):
        
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                          
                        print(path)    
                        # Create the directory  
                        # 'GeeksForGeeks' in  
                        # '/home / User / Documents'  
                        
                        print(filess[i])
                        print(shape.shape)
                        save(path_dataset+direc+"/"+filess[i]+ '.npy', shape)
                    
           
    