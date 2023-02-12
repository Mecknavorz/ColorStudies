#Color Studies :D :D
#Made by T&R (@Mecknavorz); Feb 2, 2023
#I took a butt load of photos on my neighborhood and I wanna see what's all going on w/colors and patterns
#a lil data science fun :) :)

#--------
# IMPORTS
#--------
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import math
import os


#----------
# FUNCTIONS
#----------

#generate a list of all images in a folder
def getImages():
    tbr = []
    root = input("Path to Images: ")  #the folder we wanna look in
    #should eventually redo this to make sure it only takes images
    #but there are only images in the folders I'm testing rn so nbd
    #might make it recursive eventually or at least make it scan sub directories for images
    for _, _, fnames in os.walk(root):
        for name in fnames:
            tbr.append(os.path.join(root, name))
    return tbr

#average color of the entire image
def avgColor(path):
    img = Image.open(path)
    data = np.asarray(img) #[H, W, [RGB]]
    #print("Number of Dimensions: {}".format(data.ndim))
    #print("Shape: {}".format(data.shape))
    #print(data)
    #get the average along one axis (0 =  width, 1 = height)
    avg0 = np.average(data, axis=1)
    #get the average across the remaining access to give us one RGB value
    avg1 = np.average(avg0, axis=0)
    #convert our average to an int
    #np.rint(avg1).astype(int)
    img.close()
    return avg1


#-----------------------
# CODE THAT DOES STUFF!!
#-----------------------
#get absolute paths of images we want to study
test = getImages()
#create an empty array to store the average color of each individual image
allAvgs = np.empty([0, 3])
#keeps track of index for placement and reporting progress
index = 0
#iterate over the images
for f in test:
    print("Processing: {i}/{t}".format(i=(index+1), t=len(test)))
    #get the average color of the image
    imavg = avgColor(f)
    imavg2 = np.array([imavg]) #turn it into the right shape
    allAvgs = np.concatenate((allAvgs, imavg2), axis=0) #concatenate
    #incriment the array
    index = index + 1
#return the average color of the iterated images
#print(allAvgs)
print("The average Color of these images is: {}".format(np.rint(np.average(allAvgs, axis=0))))
    
#print(test)
#avgColor(input("Path to Image: "))
