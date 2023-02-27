#Color Studies :D :D
#Made by T&R (@Mecknavorz); Feb 2, 2023
#I took a butt load of photos on my neighborhood and I wanna see what's all going on w/colors and patterns
#a lil data science fun :) :)

#--------
# IMPORTS
#--------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps
import math
import os
from pathlib import Path
import time
import pandas as pd
import cv2
import extcolors
from colormap import rgb2hex

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

#Single Color Average
def singleColorAverage(listOfFiles):
    #create an empty array to store the average color of each individual image
    allAvgs = np.empty([0, 3])
    #keeps track of index for placement and reporting progress
    index = 0
    #iterate over the images
    for f in listOfFiles:
        print("Processing: {i}/{t}".format(i=(index+1), t=len(listOfFiles)))
        #get the average color of the image
        imavg = avgColor(f)
        imavg2 = np.array([imavg]) #turn it into the right shape
        allAvgs = np.concatenate((allAvgs, imavg2), axis=0) #concatenate
        #incriment the array
        index = index + 1
    #return the average color of the iterated images
    #print(allAvgs)
    print("The average Color of these images is: {}".format(np.rint(np.average(allAvgs, axis=0))))

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

#Get the Average color relative to verticle position on the image
def HeightAverageColor(listOfFiles):
    #get the height of the image
    img = Image.open(listOfFiles[0])
    height = np.asarray(img).shape[0]
    img.close()
    #create an empty array to store the average color of each individual image's height
    allAvgs = np.empty([0, height, 3])
    index = 0
    #iterate over the images
    for f in listOfFiles:
        print("Processing: {i}/{t}".format(i=(index+1), t=len(listOfFiles)))
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

#count the # of times each color value showes up in an image
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
    img.close()
    return tbr

#rezie an image to something smaller and save it
#reduce image to % size
def shrinkImg(path, percent):
    img = Image.open(path)
    #jsut to make sure we have the input formatted right
    if(percent >= 1.0):
        percent = percent/100
    #calculate the new image size
    nwidth = int(percent * img.size[0])
    nheight = int(percent * img.size[1])
    #apply it
    img = img.resize((nwidth, nheight), Image.HAMMING)
    #save the image
    #done this way due to how I personally am storing the images
    #/source/images from a day/[images]
    #and saves them to
    #/source/resized/smol[resized images]
    fname = "\\smol_" + str(path).split('\\')[-1]
    #might need to convert path to (path(path.etc))
    saveloc = str(Path(path).parent.parent.absolute()) + "\\smol_master" + fname
    img.save(saveloc)
    img.close()
    return

#prepares a given folder to be data processed by shrinking all files and saving them to a new location
#shrinks all images to a given % size
def shrinkSet(listOfFiles, percent):
    index = 0
    #iterate over the images
    for file in listOfFiles:
        print("Processing: {i}/{t}".format(i=(index+1), t=len(listOfFiles))) #console logging for fun
        shrinkImg(file, percent) #actually shrink the images
        index += 1
    print("All Images shrunk to {p}% size; Saved to: smol_master.".format(p=percent))
    return

#generate a basic pallet from an image
def generatePallet(path):
    start = time.time()
    img = Image.open(path)
    #limit is # of colors we want
    #tolerance is how similiar colors can be
    tbr = extcolors.extract_from_image(img, tolerance = 12, limit = 12)
    img.close()
    print("Image processed in {:f} seconds".format(time.time() - start))
    index = 1
    total = 0
    '''
    for i in tbr[0]:
        percent = (i[1]/tbr[1])*100
        total += percent
        print("{j}. {c} covers {p}%".format(j=index, c=i[0], p=percent))
        index += 1
    print("for a total of {}% image coverage".format(total))
    '''
    #shape of tbr is:
    #tbr[0] is all the colors in the pallet
    #tbr[1] is the total pixel count
    #tbr[0][n] is a color and the number of pixels it shows up in
    #tbr[0][n][0] is is the rbg values (r, g, b)
    #tbr[0][n][1] is the number of pixels it shows up in
    return tbr

#graph the pallet we got into a donut
def donutPallet(pallet):
    totalpixel = pallet[1]
    colors = []
    rawpix = []
    percents = []
    text = []
    total = 0
    for i in pallet[0]:
        chex = rgb2hex(i[0][0], i[0][1], i[0][2])
        colors.append(chex) #track the color
        rawpix.append(i[1]) #track the number of pixels that are the color
        per = (i[1]/totalpixel)*100 #get percentage
        total += per #track total
        percents.append(per) #track what percent of image it is
        text.append(str(chex) + ": " + str(round(per, 1)) + "%")
    '''
    print("colors: {}".format(colors))
    print("percents: {}".format(percents))
    print("text: {}".format(text))
    print("total: {}".format(total))
    '''
    total = round(total, 2)
    #set up the chart
    explode = [0.05] * len(pallet[0])
    plt.pie(percents, colors=colors, labels=text, autopct='%1.1f%%', pctdistance=0.85, explode=explode)
    #draw circle
    center_circle = plt.Circle((0,0), 0.70, fc="white")
    fig = plt.gcf()
    #add circle into the chart
    fig.gca().add_artist(center_circle)
    #title
    plt.title("Pallet of size {n}, covering {p}%".format(n=len(pallet[0]), p=total))
    plt.show()
    return

#grabg a bunch of pallets from the entire data set
#seems to cause memory erros a lot, even with low count data, check setup and redo code here maybe
#could try running on rig to fix first problem
def allPallets(listOfFiles):
    pallets = [] #for
    index = 0
    for f in listOfFiles:
        print("Processing: {i}/{t}".format(i=(index+1), t=len(listOfFiles)))
        #generate the pallet for the image and append it
        pallets.append(generatePallet(f))
        index += 1
    return pallets

#for graphing and displaying data on a group of pallets
def graphPallets(pallets):
    #make sure it's in 3d
    #plt.style.use('_mpl-gallery')
    xs = []
    ys = []
    zs = []
    #iterate over our pallets
    for p in pallets:
        #iterate over the collors in the pallets
        for c in p[0]:
            #c[0] is the color, c[1] is the #of pixels - tested
            #append the colors to our axis
            xs.append(c[0][0])
            ys.append(c[0][1])
            zs.append(c[0][2])
    
    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(xs, ys, zs)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
    plt.show()
    print("Graph completed!")
    return

#-----------------------
# CODE THAT DOES STUFF!!
#-----------------------
if __name__ == "__main__":
    #get absolute paths of images we want to study
    test = getImages()
    #print(test)
    #donutPallet(generatePallet(test[0]))
    pallets = allPallets(test)
    graphPallets(pallets)
    #process the images
    #shrinkSet(test, .25)
    #colorCount(input("image path: "))
    #print(test)
    #avgColor(input("Path to Image: "))
