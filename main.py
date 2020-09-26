import pydicom as pyd
import numpy as np
import math
from time import sleep
from matplotlib import pyplot as plt


file_dir = '/home/ross/Documents/CODE/Radiology/Project1/data/corgi-Head-FC42'
# TODO: Implement system for reading in multitude of images based on number range

files=list()
for num in range(211,213): #because files are labeled numericly, you can input range of files you want to analyze: range(start, stop+1)
    files.append(f"{file_dir}/I{num}0000") #There are better ways to do this, but this work for now

# Which analysis you want to run:
CNR = True
NPS = True
MTF = True


# Load in each file and add it to the images 3D array (think of the images array as a stack)
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

if CNR == True: #Generates a value for every slice

    #rois of interest
    bg_roi = [1,3,0,5] #input the x and y values that define the square ROI [x1,x2,y1,y2]
    in_roi = [1,3,2,4]

    for sl, filename in enumerate(files):
        print(f"starting CNR")

        #Segment out roi's
        back_roi = np.zeros((abs(bg_roi[2]-bg_roi[3])+1,abs(bg_roi[0]-bg_roi[1])+1))
        insert_roi = np.zeros((abs(in_roi[2]-in_roi[3])+1,abs(in_roi[0]-in_roi[1])+1))

        # Copy background ROI
        for idx, i in enumerate(range(bg_roi[2],bg_roi[3]+1)):
            for idx1, j in enumerate(range(bg_roi[0],bg_roi[1]+1)):
                back_roi[idx][idx1] = images[sl][i][j] #Change first value for multi-slice use
        
        #Copy insert ROI
        for idx, i in enumerate(range(in_roi[2],in_roi[3]+1)):
            for idx1, j in enumerate(range(in_roi[0],in_roi[1]+1)):
                insert_roi[idx][idx1] = images[sl][i][j] #Change first value for multi-slice use
        

        # Average Bg_roi
        a = np.average(back_roi)
        astd = np.std(back_roi)

        # Average Ins_roi
        b = np.average(insert_roi)
        bstd = np.std(insert_roi)

        C = abs(a-b)/(0.5*(astd+bstd))
        print(f"Slice : {filename.split('/')[-1]}")
        print(f"CNR = {C}")
        print(f"Bkg_roi Average = {a}")
        print(f"Ins_roi Average = {a}")
        print(f"Bkg_roi STD = {astd}")
        print(f"Ins_roi STD = {bstd}")

if NPS:

    nps_roi = [97,220,257,400] #[x1,x2,y1,y2]

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
        print(f"Processing : {files[z].split('/')[-1]}")
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
    coeff = (ax*ay)/((abs(nps_roi[0]-nps_roi[1])+1)*(abs(nps_roi[2]-nps_roi[3])+1))


    #Get slice:
    # Copy nps ROI
    a = abs(nps_roi[0]-nps_roi[1])+1
    b = abs(nps_roi[2]-nps_roi[3])+1
    nps_slice = np.zeros((b,a))
    nps_stack = np.zeros((len(files),b,a))
    for sl in range(len(files)):
        for idx, i in enumerate(range(nps_roi[2],nps_roi[3]+1)):
            for idx1, j in enumerate(range(nps_roi[0],nps_roi[1]+1)):
                nps_slice[idx][idx1] = noise_map[sl][i][j]
    
        #Get Fourier transforms from ROI for each slice
        nps_stack[sl]=abs(np.fft.fft2(nps_slice))*abs(np.fft.fft2(nps_slice))
    
    nps_average = np.mean(nps_stack, axis=0)

    # Use fftshift in order to get a clearer graph
    test2 = np.fft.fftshift(coeff*nps_average)
    
    #Plot 2D resulting matrix with color
    fig, axi = plt.subplots(1,1)
    
    stepx = ((1/ax)/(abs(nps_roi[0]-nps_roi[1])+1))
    stepy = ((1/ay)/(abs(nps_roi[2]-nps_roi[3])+1))

    x_labels = np.arange(-(0.5/ax), (0.5/ax)+stepx, 10*stepx)
    x_ticks = np.arange(0, (abs(nps_roi[0]-nps_roi[1])+1), 10)
    plt.xticks(ticks=x_ticks, labels=x_labels)

    y_labels = np.arange(-(0.5/ay), (0.5/ay)+stepy, 10*stepy)
    y_ticks = np.arange(0, (abs(nps_roi[2]-nps_roi[3])+1), 10)
    plt.yticks(ticks=y_ticks, labels=y_labels)

    #TODO: Add colorbar
    axi.pcolor(test2,cmap="hot")
    plt.savefig("NPS.png")

    

if MTF:
    # MTF without using oversampling

    #Choose the x and y coordinates for the ends of the line you will be using (cannot be diagonal)
    esf_roi = [55,235,90,235] #[x1, y1, x2, y2] inclusive
    slice_1 = 1 #The slice from the image stack that you want to take the esf from

    #determine length of line
    if esf_roi[0] == esf_roi[2]:
        esf_len = [abs(esf_roi[1]-esf_roi[3])+1,1]
    else:
        esf_len = [abs(esf_roi[0]-esf_roi[2])+1,0]
    
    esf = np.zeros(esf_len[0])

    for idx, i in enumerate(range(esf_roi[esf_len[1]], esf_roi[esf_len[1]+2]+1)):
        #if line is along x axis
        if esf_len[1] == 1:
            esf[idx] = images[slice_1][esf_roi[0]][i]
        else:
        #if line is along y axis
            esf[idx] = images[slice_1][esf_roi[1]][i]


    #Normalize
    a = np.amax(esf)
    esf = esf/a

    #Take derivative (and find where it goes form positive to negative)
    lsf = np.diff(esf)/(dataset.PixelSpacing[0]) #denominator is the dx value, or the voxel size

    # Plot both the normalized esf and lsf on a graph
    fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
    x_values = [i for i in range(len(esf))]
    ax2.plot(x_values,esf)
    ax2.plot(x_values[:-1],lsf)
    ax2.set_xlabel("X axis")
    ax2.set_ylabel("Y axis")
    

    #Run Fourier transform on LSF to get MTF
    mtf = np.fft.fft(lsf)
    #Save figures
    x_values = [i for i in range(len(mtf))]
    ax3.plot(x_values, mtf)
    ax3.set_xlabel("X axis")

    fig2.show()
    fig2.savefig('ESF+LSF.png')

    # NOW DO OVERSAMPLED METHOD

    # ask for angle
    # ask for line location
    # ask for step size between scans

    #Calculate shifted ESF's



print('oof')
