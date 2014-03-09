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


for d in dirnames:
    print d
    filenames= []
    for (dirpath, dirnames, filenames) in walk('../londiani/%s'%d):
        filenames = sorted(filenames)
            
        
        i = []
        for f in filenames:
            print f
            if f[-3:] =='hdr':
                print f
                
                img = envi.open('../londiani/%s/%s'%(d,f))
                l = img.load()
                m = img.open_memmap()
                i.append(m)
        #got i now process
        print len(i)
        if len(i) == 6:
            big = np.dstack((i[0],i[1],i[2],i[3],i[4],i[5]))
        if len(i) == 5:
            big = np.dstack((i[0],i[1],i[2],i[3],i[4]))
        if len(i) == 4:
            big = np.dstack((i[0],i[1],i[2],i[3])



        pc = principal_components(big)
        pc_090 = pc.reduce(fraction=0.99)
        big_pc = pc_090.transform(big)
        #v = imshow(big_pc)
        #v = imshow(big_pc, stretch_all=True)

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
                save_rgb('../londiani/images/%s_%s.jpg'%(d,i),pcmm==k)
                print '../londiani/images/%s_%s.jpg'%(d,i)
                i = i +1

        start = 220

        for j in [1,2,3,4]:
            plt.suptitle('PCA with KMeans Clustering for %s'%d, fontsize=12)
            plt.subplot(start +j),plt.imshow((pcmm==pcclusters[j-1][0])==0,cmap = 'gray')
            plt.title('PCA Cluster %s'%j), plt.xticks([]), plt.yticks([]),plt.savefig('../londiani/images/%s_pc_comp.jpg'%d)

        

        
