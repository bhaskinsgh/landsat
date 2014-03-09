import time
from spectral import *
import spectral.io.envi as envi
img3 = envi.open('./8Dec2013/band3.hdr')
l = img3.load()
m = img3.open_memmap()
s3 = m.astype('uint8')


img4 = envi.open('./8Dec2013/band4.hdr')
l4 = img4.load()
m4 = img4.open_memmap()
s4 = m4.astype('uint8')

img5 = envi.open('./8Dec2013/band5.hdr')
l5 = img5.load()
m5= img5.open_memmap()
s5 = m5.astype('uint8')

import numpy as np

rgb= np.dstack((s5,s4,s3))

v = imshow(rgb)
time.sleep(3)










