import cv2
from os import walk
import time
from spectral import *
import spectral.io.envi as envi
import numpy as np

dirnames = []
for (dirpath, dirnames, filenames) in walk('../londiani'):
    dirnames = sorted(dirnames)
    break

print dirnames
img1 = cv2.imread('classes_new_%s.jpg'%dirnames[5])
height , width , layers =  img1.shape
print height, width
video = cv2.VideoWriter('video.avi',-1,1,(width,height))

try:
	for d in dirnames:
		if d <> 'images':
			print d
			img1 = cv2.imread('classes_new_%s.jpg'%d)
			video.write(img1)
			img1 = None
except:
	pass

cv2.destroyAllWindows()
video.release()
