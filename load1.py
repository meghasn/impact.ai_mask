import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

img_array = np.load('megha.npy')
img_array2 = np.load('megha_mask.npy')
img_array3 = np.load('akash.npy')
img = img_array.flatten() 
img2 = img_array2.flatten() 

img3=img_array3.flatten()
print(img.shape)
x=cosine_similarity([img],[img2,img3])
# result=cosine_similarity(img,img2)
# result = spatial.procrustes([img_array],[img_array2,img_array3])[-1]
# result2 = spatial.procrustes(img_array2,img_array3)[-1]
# result2=1 - spatial.distance.cosine(img , img3)
# result3=1 - spatial.distance.cosine(img2 , img3)
# print(img_array)
# print(img_array2)
print(x)
# print(result2)
# print(result3)

