import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import pydicom
import datetime
from datetime import datetime
import cv2
import time
from skimage import exposure
import ast

# read data file 
csv_path = 'data_csv.csv'
new_df = pd.read_csv(csv_path, delimiter=';', dtype={'sourcefile':str})

# function for displaying image arrays
def vis_array(img):
    plt.figure()
    plt.imshow(img, 'gray')
    plt.pause(0.05)

# function to locate the center of mass    
def new_cropping_single_dist(img):
    org_img = img
    opening_it = 5
    kernel_size = (25, 25)
    kernel = np.ones(kernel_size, np.uint8)
  
    _, img = cv2.threshold(img, img.min(), img.max(), cv2.THRESH_BINARY)
  
    img = cv2.GaussianBlur(img, kernel_size, 0)
    img = cv2.dilate(img, kernel, 5)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opening_it)
    opening = cv2.copyMakeBorder(opening, top=1, bottom=1, left=1, right=1, borderType= cv2.BORDER_CONSTANT, value=0 )
    opening = opening.astype(np.uint8)
  
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, kernel_size, 0)
    maxLoc = cv2.minMaxLoc(dist)[-1]
    return maxLoc

# function to add padding or perform cropping 
# to make the center of mass be the center of the image 
def pad_or_crop_single_maxloc(img, maxloc):
    w = img.shape[1]
    h = img.shape[0]

    if maxloc[0] >= w-maxloc[0]:
        img_new = np.full((h,2*maxloc[0]), 0.0)
        img_new[:, :w] = img
    elif maxloc[0] < w-maxloc[0]:
        img_new = np.full((h,2*(w-maxloc[0])), 0.0)
        img_new[:, 2*(w-maxloc[0])-w:] = img

    img = img_new
    w = img_new.shape[1]
    h = img_new.shape[0]

    if maxloc[1] >= h-maxloc[1]:
        img_new = np.full((2*maxloc[1],w), 0.0)
        img_new[:h, :] = img
    elif maxloc[1] < h-maxloc[1]:
        img_new = np.full((2*(h-maxloc[1]),w), 0.0)
        img_new[2*(h-maxloc[1])-h:, :] = img

    img = img_new
    w = img.shape[1]
    h = img.shape[0]

    if h > full_height:
        img = img[int((h-full_height)/2):int((h-full_height)/2)+full_height,:]
    if w > full_width:
        img = img[:, int((w-full_width)/2):int((w-full_width)/2)+full_width]
    img_new = np.full((full_height,full_width), 0.0)
    img_new[int((img_new.shape[0]-img.shape[0])/2):int((img_new.shape[0]-img.shape[0])/2)+img.shape[0], int((img_new.shape[1]-img.shape[1])/2):int((img_new.shape[1]-img.shape[1])/2)+img.shape[1]] = img
    return img_new

count = 0
# define whether image arrays are to be displayed
if_vis = False
# define whether only a subset of images is to be processed 
if_notall = False
# define the image height and width
full_height = 4096
full_width = 3328
# define the starting image index
start_index = 0
# define the folder for preprocessed images
new_path_folder = 'data_preprocessed/'

for i in range(start_index, new_df.shape[0]):
    start  = time.time()
    row = new_df.loc[i,:]
    if_train = int(row['if_train'])
    if_val = int(row['if_val'])
    if_test = int(row['if_test'])
    local_path = row['full_path']
    basename = row['basename'][:-3]+'png'
    dicom_imagelaterality = row['dicom_imagelaterality']
    
    if (if_train or if_val or if_test):
        print(i)
        dicom_windowcenter = str(row['dicom_windowcenter']).strip('][').replace('\'','').split(', ') 
        dicom_windowwidth = str(row['dicom_windowwidth']).strip('][').replace('\'','').split(', ') 
        dicom_windowcenter = [int(float(value)) for value in dicom_windowcenter]
        dicom_windowwidth = [int(float(value)) for value in dicom_windowwidth]
        
        img_dcm = pydicom.dcmread(local_path)
        new_path = new_path_folder + basename
        # keep a record of currupted images
        try:
          img_array = img_dcm.pixel_array
        except:
          print('dicom_problem')
          new_df.loc[i, 'dicom_corrupted'] = 1
          continue
      
        # display image arrays if it is needed
        if if_vis:
            vis_array(img_array)
            
        # flip the image if necceary to make all breasts left-posed
        if row['dicom_imagelaterality']  == 'R':
            img_array = cv2.flip(img_array, 1) 
        if if_vis:
            vis_array(img_array)
        
        # intensity rescaling according to window center and window width
        img_array = exposure.rescale_intensity(img_array, in_range=(dicom_windowcenter[0]-dicom_windowwidth[0]/2, dicom_windowcenter[0]+dicom_windowwidth[0]/2))
        if if_vis:
            vis_array(img_array)

        # invert the color if needed
        if row['dicom_photometricinterpretation']== 'MONOCHROME1':
            img_array = cv2.bitwise_not(img_array)
        if if_vis:
            vis_array(new_img)
        
        # crop with distance transform and pad 
        maxLoc = new_cropping_single_dist(img_array)
        # keep a record of the center of the mass in original coordinate space
        new_df.loc[i, 'dicom_maxLoc'] = str(maxLoc)
        new_img = pad_or_crop_single_maxloc(img_array, maxLoc) 
        if if_vis:
            vis_array(new_img)
    
        # save preprocessed images    
        cv2.imwrite(new_path, new_img.astype(np.uint16))     
        count += 1 
        
        # stop if we choose to process only a subset of images
        if if_notall:
            if count > 10:
                break

        # save data csv after processing every one hundred images
        if count % 100 == 0:
            new_df.to_csv('data_csv.csv', sep=';', index=False)
        print(str(count), ' out of ', str(new_df.shape[0]), ' images are done')
        print('time for processing: ', str(time.time()-start))

new_df.to_csv('data_csv.csv', sep=';', index=False)

