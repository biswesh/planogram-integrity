import scipy.misc
import numpy as np
import cv2


imgPath = '/home/ubuntu/instanceMatch/data19/15:48:022017-02-192edf9c02fb50c9fc03453faaf124163ac3243028.jpg' 
img = cv2.imread(imgPath)
#img_array =  np.transpose(img, (1, 0, 2))
#print(img.shape)
#print(img_array.shape)

img_array = cv2.flip(img,0)
#scipy.misc.imsave('rgb_gradient.png', img_array)
print(img_array.shape)
cv2.imshow('image',img_array)
