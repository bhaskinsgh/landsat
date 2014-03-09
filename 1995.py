import time
from spectral import *
import spectral.io.envi as envi
import numpy as np


i = []
names = [1,2,3,4,5,7]
for n in names:
        print n
        print './21jan1995/band%s.hdr'%n
	img = envi.open('./21Jan1995/band%s.hdr'%n)
	l = img.load()
	m = img.open_memmap()
	i.append(m)


big = np.dstack((i[0],i[1],i[2],i[3],i[4],i[5]))
#rgb= np.dstack((s5,s4,s3))

(mm,c) = kmeans(big,20,20)
clusters = {}
for i in range(20):
    clusters[i] = np.sum(mm==i)

print clusters
a =sorted(clusters.items(), key=lambda clusters: clusters[1])
a.reverse()
clusters = a


pc = principal_components(big)
pc_099 = pc.reduce(fraction=0.99)
big_pc = pc_099.transform(big)
v = imshow(big_pc)
v = imshow(big_pc, stretch_all=True)

(pcmm,c) = kmeans(big_pc,10,20)
pcclusters = {}
for i in range(20):
    pcclusters[i] = np.sum(pcmm==i)

print pcclusters
a =sorted(pcclusters.items(), key=lambda pcclusters: pcclusters[1])
a.reverse()
pcclusters = a
i =1
for k,v in pcclusters:
    if v > 0:
    	save_rgb('1995_%s.jpg'%i,pcmm==k)
        i = i +1








    










