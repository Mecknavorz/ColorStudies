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
from pathlib import Path
import time

#----------
# FUNCTIONS
#----------

#generate a list of all images in a folder
def getImages():
    tbr = []
    root = input("Path to Images: ")  #the folder we wanna look in
    #might make it recursive eventually or at least make it scan sub directories for images
    for _, _, fnames in os.walk(root):
        for name in fnames:
            if name.endswith(('.jpg', '.png', 'jpeg')):
                tbr.append(os.path.join(root, name))
    return tbr

#average color of the entire image
def avgColor(path):
    img = Image.open(path)
    data = np.asarray(img) #[H, W, [RGB]]
    #print("Number of Dimensions: {}".format(data.ndim))
    #print("Shape: {}".format(data.shape))
    #print(data)
    #get the average along one axis (0 returns width size, 1 returns height size)
    avg0 = np.average(data, axis=1)
    #get the average across the remaining access to give us one RGB value
    avg1 = np.average(avg0, axis=0)
    #convert our average to an int
    #np.rint(avg1).astype(int)
    img.close()
    return avg1

#average colors of the image along 
def avgColorH(path):
    img = Image.open(path)
    data = np.asarray(img) #[H, W, [RGB]]
    #print("Number of Dimensions: {}".format(data.ndim))
    #print("Shape: {}".format(data.shape))
    #print(data)
    #get the average along one axis (0 =  width, 1 = height)
    avg0 = np.average(data, axis=1)
    img.close()
    return avg0

#count the #of times each color value showes up in an image
#slow, might not even be what I need
#takes about 33 seconds for 1 12MP image :/ :/
def colorCount(path):
    img = Image.open(path)
    data = np.asarray(img)
    tbr = np.zeros([3, 256]) #our 3 colors and the possible values they can have 0-255
    #I'm not sure if there's a faster way to do this other than checking each pixel individually
    start = time.time()
    for w in data:
        for c in w:
            #increment the count for each of the colors
            tbr[0, c[0]] += 1
            tbr[1, c[1]] += 1
            tbr[2, c[2]] += 1
    print("Image processed in {:f} seconds".format(time.time() - start))
    return tbr

#-----------------------
# CODE THAT DOES STUFF!!
#-----------------------
#get absolute paths of images we want to study
test = getImages()
#print(test)
#colorCount(input("image path: "))

'''#Get the Average color relative to verticle position on the image
#get the height of the image
img = Image.open(test[0])
height = np.asarray(img).shape[0]
img.close()
#create an empty array to store the average color of each individual image's height
allAvgs = np.empty([0, height, 3])
index = 0
#iterate over the images
for f in test:
    print("Processing: {i}/{t}".format(i=(index+1), t=len(test)))
    #get the average color of the image
    imavg = avgColorH(f)
    imavg2 = np.array([imavg]) #turn it into the right shape
    allAvgs = np.concatenate((allAvgs, imavg2), axis=0) #concatenate
    #incriment the array
    index = index + 1
#return the average color of the iterated images
#print(allAvgs)
print("Average Color per height generated! Created output image in working directory.")
havg = np.rint(np.average(allAvgs, axis=0)).astype(np.uint8) #make the values all neat and get the average
havgImg = np.tile(havg, ((havg.shape[0] // 10), 1, 1)) #create an array of image size which we can turn into a picture
out = Image.fromarray(havgImg)
#determine where to save
out.save("Average Color by Height.jpg")
'''

'''#Single Color Average
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
'''
#print(test)
#avgColor(input("Path to Image: "))
