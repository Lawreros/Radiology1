import pydicom as pyd
import numpy as np
import math
from time import sleep
from matplotlib import pyplot as plt


file_dir = '/home/ross/Documents/CODE/Radiology/Project1/data/corgi-Head-FC42'
# TODO: Implement system for reading in multitude of images based on number range

files=list()
for num in range(211,213): #input range of files you want to analyze: range(start, stop+1)
    files.append(f"{file_dir}/I{num}0000")


CNR = True
NPS = True
MTF = True

#rois of interest
bg_roi = [1,3,0,5] #input for corners of square ROI [x1,x2,y1,y2]
in_roi = [1,3,2,4]


for idx, dicom in enumerate(files):
    dataset = pyd.dcmread(dicom)
    print(f'The pixel sizes for this image are: {dataset.PixelSpacing[0]} mm by {dataset.PixelSpacing[1]} mm')

    print(f"Loading in image pixel array...")
    if idx == 0:
        dim = np.array(dataset.pixel_array)
        print(f'Dimensions: {dim.shape[0]}, {dim.shape[1]}')
        images = np.zeros((len(files),dim.shape[0],dim.shape[1]))
        images[idx] = dim
    else:
        images[idx] = dataset.pixel_array

if CNR == True: #Currently only looking at first slice
    print(f"starting CNR")

    #Segment out roi's
    back_roi = np.zeros((abs(bg_roi[2]-bg_roi[3])+1,abs(bg_roi[0]-bg_roi[1])+1))
    insert_roi = np.zeros((abs(in_roi[2]-in_roi[3])+1,abs(in_roi[0]-in_roi[1])+1))

    # Copy background ROI
    for idx, i in enumerate(range(bg_roi[2],bg_roi[3]+1)):
        for idx1, j in enumerate(range(bg_roi[0],bg_roi[1]+1)):
            back_roi[idx][idx1] = images[0][i][j] #Change first value for multi-slice use
    
    #Copy insert ROI
    for idx, i in enumerate(range(in_roi[2],in_roi[3]+1)):
        for idx1, j in enumerate(range(in_roi[0],in_roi[1]+1)):
            insert_roi[idx][idx1] = images[0][i][j] #Change first value for multi-slice use
    

    # Average Bg_roi
    a = np.average(back_roi)
    astd = np.std(back_roi)

    # Average Ins_roi
    b = np.average(insert_roi)
    bstd = np.std(insert_roi)

    C = abs(a-b)/(0.5*(astd+bstd))
    print(f"CNR = {C}")
    print(f"Bkg_roi Average = {a}")
    print(f"Ins_roi Average = {a}")
    print(f"Bkg_roi STD = {astd}")
    print(f"Ins_roi STD = {bstd}")

if NPS:
    print('Begining NPS calculations')
    #Create Noise Map
    dim = images.shape

    #Make empty noise map to fill with values
    noise_map = np.zeros((dim[0],dim[1],dim[2]))

    #Create a 5x5 kernel that walks through the images and creates a noise map voxel equivalent
    kernel = [5,5] #kernel [x,y]

    #convert to a usable halves
    kernel[0] = math.floor(kernel[0]/2)
    kernel[1] = math.floor(kernel[1]/2)

    for z in range(dim[0]):
        for y in range(dim[1]):
            for x in range(dim[2]):
                # Account for fringe cases where the kernel exceeds the image
                if x-kernel[0] < 0 and y-kernel[1] < 0:
                    noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][0:y+kernel[1]+1, 0:x+kernel[0]+1])
                elif x+kernel[0] > dim[2] and y+kernel[1] > dim[1]:
                    noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][y-kernel[1]:dim[1], x-kernel[0]:dim[2]])
                elif x-kernel[0] < 0:
                    noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][y-kernel[1]:y+kernel[1]+1, 0:x+kernel[0]+1]) 
                elif y-kernel[1] < 0:
                    noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][0:y+kernel[1]+1, x-kernel[0]:x+kernel[0]+1]) 
                elif x+kernel[0] >= dim[2]:
                     noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][y-kernel[1]:y+kernel[1]+1, x-kernel[0]:dim[0]])
                elif y+kernel[1] >= dim[1]:
                    noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][y-kernel[1]:dim[1], x-kernel[0]:x+kernel[0]+1])
                # Now for when everything is good...
                else:
                    noise_map[z][y][x] = images[z][y][x] - np.mean(images[z][y-kernel[1]:y+kernel[1]+1, x-kernel[0]:x+kernel[0]+1])

    #TODO: Add creation of test images to make sure that the noise map makes sense

    # With our noise map created, we can now calculate the Noise-Power Spectrum
    # Voxel size in x and y direction
    ax = dataset.PixelSpacing[0]
    ay = dataset.PixelSpacing[1]
    
    #Calculate coefficients for NPS calculation
    nps_roi = [97,220,257,400] #[x1,x2,y1,y2]
    coeff = (ax*ay)/((abs(nps_roi[0]-nps_roi[1])+1)*(abs(nps_roi[2]-nps_roi[3])+1))

    #Get Fourier transforms from ROI for each slice

    #Get slice:
    # Copy nps ROI
    a = abs(nps_roi[0]-nps_roi[1])+1
    b = abs(nps_roi[2]-nps_roi[3])+1
    nps_slice = np.zeros((b,a))
    nps_stack = np.zeros((len(files),b,a))
    for idx, i in enumerate(range(nps_roi[2],nps_roi[3]+1)):
        for idx1, j in enumerate(range(nps_roi[0],nps_roi[1]+1)):
            nps_slice[idx][idx1] = noise_map[0][i][j] #Change first value for multi-slice use
    
    nps_stack[0]=coeff*abs(np.fft.fft2(nps_slice))*abs(np.fft.fft2(nps_slice))
    nps_stack[0] = np.fft.fftshift(nps_stack[0])
    
    #Plot 2D resulting matrix with color
    fig, ax = plt.subplots(1,1)
    ax.pcolor(nps_stack[0])
    plt.savefig("test.png")

    

if MTF:
    # Select line
    pass

print('oof')
