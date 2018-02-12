
# coding: utf-8

# In[20]:

import os

path = "/home/aw/"
os.chdir(path)
currDir = os.getcwd()

print(currDir)

name = "ban.jpg"

import cv2
print("\n\tcv2.imread(name, 0)")
cv2image = cv2.imread(name, 0)
print(type(cv2image))
print(cv2image.shape)
print(cv2image[0][0])


import matplotlib
from matplotlib import pyplot as plt

print("\n\tplt.imread(name, 0)")
pltimage = plt.imread(name, 0)
print(type(pltimage))
print(pltimage.shape)
print(pltimage[0][0])

print("\n\tcv2.cvtColor(pltimage, cv2.COLOR_BGR2GRAY)")
cv2image2 = cv2.cvtColor(pltimage, cv2.COLOR_BGR2GRAY)
print(type(cv2image2))
print(cv2image2.shape)
print(cv2image2[0][0])

#testing COLOR_"RGB"
print("\n\tcv2.cvtColor(pltimage, cv2.COLORBGR2GRAY)")
cv2image3 = cv2.cvtColor(pltimage, cv2.COLOR_RGB2GRAY)
print(type(cv2image3))
print(cv2image3.shape)
print(cv2image3[0][0])

# does zero in plt make difference? let's find out
print("\n\tplt.imread(name)")
plt2image = plt.imread(name)
print(type(plt2image))
print(plt2image.shape)
print(plt2image[0][0])


# In[ ]:



