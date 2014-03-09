import time
from spectral import *
import spectral.io.envi as envi
import numpy as np
from os import walk
from matplotlib import pyplot as plt

im = []
d = '8Dec2013'
d='20131208'
maxclasses = 20
def getdatesimage(dt):
	print 'Getting %s'%dt
	for (dirpath, dirnames, filenames) in walk('../londiani/%s'%dt):
		filenames = sorted(filenames)
			
		
		im = []
		for f in filenames:
			#print f
			if f[-3:] =='hdr':
				
				img = envi.open('../londiani/%s/%s'%(dt,f))
				l = img.load()
				m = img.open_memmap()
				im.append(m)

	i = im
	#print len(im)
	if len(im) == 7:
		big = np.dstack((im[0],im[1],im[2],im[3],im[4],im[5],im[6]))
	if len(im) == 6:
		big = np.dstack((im[0],im[1],im[2],im[3],im[4],im[5]))
	if len(im) == 5:
		big = np.dstack((im[0],im[1],im[2],im[3],im[4]))
	if len(im) == 4:
		big = np.dstack((im[0],im[1],im[2],im[3]))
		
	return big
	
big = getdatesimage(d)
#rgb= np.dstack((s5,s4,s3))
savekmeans = False
if savekmeans:
	(mm,c) = kmeans(big,maxclasses,10)
	clusters = {}
	for i in range(maxclasses):
		clusters[i] = np.sum(mm==i)

	print clusters
	a =sorted(clusters.items(), key=lambda clusters: clusters[1])
	a.reverse()
	clusters = a
	i = 1

	for k,v in clusters:
		if v > 0:
			save_rgb('../londiani/images/%s_%s_k.jpg'%(d,i),mm==k)
			print '../londiani/images/%s_%s_k.jpg'%(d,i)
			i = i +1

pc = principal_components(big)
pc_099 = pc.reduce(fraction=0.99)
big_pc = pc_099.transform(big)
#v = imshow(big_pc)
#v = imshow(big_pc, stretch_all=True)
'''
hist, bins = np.histogram(pcmm)
import matplotlib.pyplot as plt
center = (bins[:-1] + bins[1:])/2
plt.bar(center,hist)
'''

(pcmm,c) = kmeans(big_pc,maxclasses,10)
pcclusters = {}
for i in range(maxclasses):
    pcclusters[i] = np.sum(pcmm==i)


a =sorted(pcclusters.items(), key=lambda pcclusters: pcclusters[1])
a.reverse()
pcclusters = a

i =1
j = 1
start = 220

for j in [1,2,3,4]:
    plt.suptitle('PCA with KMeans Clustering for %s'%d, fontsize=12)
    plt.subplot(start +j),plt.imshow((pcmm==pcclusters[j-1][0])==0,cmap = 'gray')
    plt.title('PCA Cluster %s'%j), plt.xticks([]), plt.yticks([]),plt.savefig('../londiani/images/%s_pc_comp.jpg'%d)


saverawclusters = False
if saverawclusters:
	for k,v in pcclusters:
		if v > 0:
			save_rgb('2013_%s.jpg'%i,pcmm==k)
			print '2013_%s.jpg'
			i = i +1
newarray = []
for j in [1,2,3]:
	newarray.append(pcmm==pcclusters[j-1][0])

newbig = np.dstack((newarray[0],newarray[1],newarray[2]))
cols = spectral.spy_colors
newcols = np.array([cols[1],cols[6],cols[5],cols[7],cols[3], cols[2], cols[1],cols[19],cols[18]])
saveclassimage = True
if saveclassimage:
	save_image('classes_%s.jpg'%d,pcmm,colors=spectral.spy_colors)
keys = []
for data in pcclusters:
	keys.append(data[0])
pcnew = pcmm.copy()
try:
	for i in range(maxclasses):
		k = keys[i]
		pcnew[pcnew==k] = i
		
	
except:
	pass
if saveclassimage:
	save_image('classes_new_%s.jpg'%d,pcnew,colors=spectral.spy_colors)
	

save_image('true_color_%s.jpg'%d,big,(3,2,1))
	
print 'starting Supervised classification'
from PIL import Image
img = Image.open('gt2.png')
pix = np.array(img)
imshow(pix)
(m, c) = kmeans(pix, 8, 30)

d ='2013'
classes = create_training_classes(big_pc,m)
gmlc = GaussianClassifier(classes)
clmap = gmlc.classify_image(big_pc)
#v = imshow(classes=clmap)
class_2013_pc = np.copy(clmap)
#save_image('class_train_pc_%s.jpg'%d,clmap,colors=spectral.spy_colors)


classes = create_training_classes(big,m)
gmlc = GaussianClassifier(classes)
clmap = gmlc.classify_image(big)
#v = imshow(classes=clmap)
class_2013_orig = np.copy(clmap)
#save_image('class_train_orig_%s.jpg'%d,clmap,colors=spectral.spy_colors)

'''
d='1995'
print 'starting PCA %s'%d
img1995 = getdatesimage('19950121')
pc = principal_components(img1995)
pc_099 = pc.reduce(fraction=0.99)
img1995_pc = pc_099.transform(img1995)

print 'start training on %s PCA'%d
classes = create_training_classes(img1995_pc,m)
gmlc = GaussianClassifier(classes)
clmap = gmlc.classify_image(img1995_pc)
#v = imshow(classes=clmap)
class_1995_pc = np.copy(clmap)
#save_image('class_train_pc_%s.jpg'%d,clmap,colors=spectral.spy_colors)

print 'start training on %s Original'%d
classes = create_training_classes(img1995,m)
gmlc = GaussianClassifier(classes) #for some reason this is broken here!!
clmap = gmlc.classify_image(img1995)
class_1995_orig = np.copy(clmap)
#v = imshow(classes=clmap)
#save_image('class_train_orig_%s.jpg'%d,clmap,colors=spectral.spy_colors)

d ='2013'
classes = create_training_classes(big_pc,m)
gmlc = GaussianClassifier(classes)
clmap = gmlc.classify_image(big_pc)

h2013,b2013 = np.histogram(class_2013_pc, bins=range(10))
h1995,b1995 = np.histogram(class_1995_pc, bins=range(10))

forest1995 = h1995[1]+ h1995[3]
forest2013 = h2013[1]+ h2013[2] +h2013[3]

images = [class_2013_pc, class_1995_pc]
titles = ['2013 PCA', '1995 PCA']
images = [class_2013_pc, class_1995_pc]
titles = ['2013 PCA', '1995 PCA']
start = 120

for j in [1,2]:
	plt.suptitle('Results for Trained Classification 2013 - 1995', fontsize=12)
	plt.subplot(start +j),plt.imshow(images[j-1],cmap = 'gray')
	plt.title(titles[j-1]), plt.xticks([]), plt.yticks([]),plt.savefig('class_comp.jpg')
	
	

'''





