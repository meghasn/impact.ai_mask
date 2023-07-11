from imutils import face_utils
import dlib
import cv2
import numpy as np
from numpy import save
import os
from mask import create_mask
path_to_read="C:/Users/pnc/Documents/impact.ai/mask-feature-exp/lfw-deepfunneled-2/"
path="C:/Users/pnc/Documents/impact.ai/mask-feature-exp/lfw-deepfunneled-mask/"
    
directory = os.path.join(path_to_read)
for root,dirs,files in os.walk(directory):
    for direc in dirs:
        fil=os.path.join(directory + direc)
        path1 = os.path.join(path + direc)
        os.mkdir(path1)
        path2 = os.path.join(path1 +"/"+ "test")
        os.mkdir(path2)
        path3 = os.path.join(path1 +"/"+ "train")
        os.mkdir(path3)
        # print(direc)
        for toots,dirss,filess in os.walk(fil):
                le=len(filess)
                for i in range(len(filess)):
                    
                    image = cv2.imread(fil + "/" + filess[i])
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if(le==1):
                        print(direc)
                        print(path1)
                        os.rmdir(path2)
                        os.rmdir(path3)
                        os.rmdir(path1)
                    if(i<le/2 and le!=1):
                        
                        create_mask(fil + "/" + filess[i], filess[i], path2)
                    if(i>=le/2 and le!=1):
                       
                        create_mask(fil + "/" + filess[i], filess[i], path3)
                        
                    print(filess[i])
                    print(path1)
                    # create_mask(fil + "/" + filess[i], filess[i], path1)
                    