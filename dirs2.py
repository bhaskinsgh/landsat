from os import walk
import time
from spectral import *
import spectral.io.envi as envi
import numpy as np
from matplotlib import pyplot as plt

maxclasses = 20
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
            big = np.dstack((i[0],i[1],i[2],i[3]))
        #Do kmeans clustering on original image
        dokmeans = False
        if dokmeans:
            (mm,c) = kmeans(big,maxclasses,20)
            clusters = {}
            for i in range(maxclasses):
                clusters[i] = np.sum(mm==i)

            print clusters
            a =sorted(clusters.items(), key=lambda clusters: clusters[1])
            a.reverse()
            clusters = a
                    
        pc = principal_components(big)
        pc_099 = pc.reduce(fraction=0.99) #Try to keep 99% of variance
        big_pc = pc_099.transform(big)
        #v = imshow(big_pc)
        #v = imshow(big_pc, stretch_all=True)

        (pcmm,c) = kmeans(big_pc,maxclasses,20) #10 clusters 20 iterations
        pcclusters = {}
        for i in range(maxclasses):
            pcclusters[i] = np.sum(pcmm==i) #find how many pixels in each cluster

        print pcclusters
        a =sorted(pcclusters.items(), key=lambda pcclusters: pcclusters[1]) # sort
        a.reverse() #sort descending
        pcclusters = a
        i =1
        save = False
        if save:
            for k,v in pcclusters:
                if v > 0:
                    save_rgb('../londiani/images/%s_%s.jpg'%(i,d),pcmm==k)
                    print '../londiani/images/%s_%s.jpg'%(i,d)
                    i = i +1

        start = 220 #2 rows, 2 cols , 
        savepca = False
        if savepca:
            for j in [1,2,3,4]:
                plt.suptitle('PCA with KMeans Clustering for %s'%d, fontsize=12)
                plt.subplot(start +j),plt.imshow((pcmm==pcclusters[j-1][0])==0,cmap = 'gray')
                plt.title('PCA Cluster %s'%j), plt.xticks([]), plt.yticks([]),plt.savefig('../londiani/images/pc_comp_%s.jpg'%d)
        
        plt.clf() #clear the figure
        plt.cla() #to be sure, to be sure
        save_image('../londiani/images/pc_class_%s.jpg'%d,pcmm,colors= spectral.spy_colors)
        keys = []
        for data in pcclusters:
            keys.append(data[0])
        print pcmm
        pcnew = pcmm.copy()
        try:
            for i in range(maxclasses):
                k = keys[i]
                print k,i
                pcnew[pcnew==k] = i
                
            
        except:
            pass
        saveclassimage = True
        if saveclassimage:
            save_image('classes_new_%s.jpg'%d,pcnew,colors=spectral.spy_colors)
                

        

        
