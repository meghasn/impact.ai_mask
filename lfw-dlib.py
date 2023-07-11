import cv2
import sys
import os
import dlib
imagePath ="C:/Users/pnc/Documents/impact.ai/mask-feature-exp/lfw-deepfunneled/"
directory = os.path.join(imagePath)
a=0
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
for root,dirs,files in os.walk(directory):
    for direc in dirs:
        fil=os.path.join(directory + direc)
        
        for toots,dirss,filess in os.walk(fil):
                print(direc)
                if(len(filess)==0):
                    os.rmdir(fil)
                for i in range(len(filess)):
                    a=a+1
                    
                    image = cv2.imread(fil + "/" + filess[i])
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray, 0)
                    # print(filess[i])
                    # print("detect")
                    # print(len(rects))
    

                    h=len(rects)
                    # print(h)
                    if(h != 1):
                        # print(filess[i])
                        os.remove(fil + "/" + filess[i])
                    
# print(a)            