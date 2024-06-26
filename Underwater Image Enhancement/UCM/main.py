import os
import numpy as np
import cv2
import natsort
import xlwt
import datetime
from skimage.color import rgb2hsv, hsv2rgb
from multiprocessing import Pool
import traceback

###
### color_equalisation.py
###
def cal_equalisation(img, ratio):
    Array = img * ratio
    Array = np.clip(Array, 0, 255)
    return Array


def RGB_equalisation(img):
    img = np.float32(img)
    avg_RGB = []
    for i in range(3):
        avg = np.mean(img[:, :, i])
        avg_RGB.append(avg)
    # print('avg_RGB',avg_RGB)
    a_r = avg_RGB[0] / avg_RGB[2]
    a_g = avg_RGB[0] / avg_RGB[1]
    ratio = [0, a_g, a_r]
    for i in range(1, 3):
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
    return img


##
## global_histogram_stretching.py
##
def histogram_r(r_array, I_min, I_max):
    array_Global_histogram_stretching = np.zeros_like(r_array)
    mask1 = r_array < I_min
    mask2 = r_array > I_max
    array_Global_histogram_stretching[mask1] = I_min
    array_Global_histogram_stretching[mask2] = 255
    mask3 = ~(mask1 | mask2)
    array_Global_histogram_stretching[mask3] = (
        (r_array[mask3] - I_min) * ((255 - I_min) / (I_max - I_min))
    ) + I_min
    return array_Global_histogram_stretching


def histogram_g(r_array, I_min, I_max):
    array_Global_histogram_stretching = np.zeros_like(r_array)
    mask1 = r_array < I_min
    mask2 = r_array > I_max
    array_Global_histogram_stretching[mask1] = 0
    array_Global_histogram_stretching[mask2] = 255
    mask3 = ~(mask1 | mask2)
    array_Global_histogram_stretching[mask3] = (r_array[mask3] - I_min) * (
        (255) / (I_max - I_min)
    )
    return array_Global_histogram_stretching


def histogram_b(r_array, I_min, I_max):
    array_Global_histogram_stretching = np.zeros_like(r_array)
    mask1 = r_array < I_min
    mask2 = r_array > I_max
    array_Global_histogram_stretching[mask1] = 0
    array_Global_histogram_stretching[mask2] = I_max
    mask3 = ~(mask1 | mask2)
    array_Global_histogram_stretching[mask3] = (r_array[mask3] - I_min) * (
        (I_max) / (I_max - I_min)
    )
    return array_Global_histogram_stretching


def stretching(img):
    height, width, _ = img.shape
    length = height * width

    R_rray = img[:, :, 2].flatten()
    R_rray.sort()
    I_min_r = int(R_rray[int(length / 500)])
    I_max_r = int(R_rray[-int(length / 500)])

    G_rray = img[:, :, 1].flatten()
    G_rray.sort()
    I_min_g = int(G_rray[int(length / 500)])
    I_max_g = int(G_rray[-int(length / 500)])

    B_rray = img[:, :, 0].flatten()
    B_rray.sort()
    I_min_b = int(B_rray[int(length / 500)])
    I_max_b = int(B_rray[-int(length / 500)])

    img[:, :, 2] = histogram_r(img[:, :, 2], I_min_r, I_max_r)
    img[:, :, 1] = histogram_g(img[:, :, 1], I_min_g, I_max_g)
    img[:, :, 0] = histogram_b(img[:, :, 0], I_min_b, I_max_b)

    return img


##
## hsvStretching.py
##
def HSVStretching(sceneRadiance):
    sceneRadiance = np.uint8(sceneRadiance)
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    
    img_s_stretching = global_stretching(s)
    img_v_stretching = global_stretching(v)
    
    labArray = np.zeros_like(img_hsv, dtype="float64")
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255
    
    return img_rgb


##
## global_Stretching.py
##
def global_stretching(img_L):
    I_min = np.min(img_L)
    I_max = np.max(img_L)
    
    array_Global_histogram_stretching_L = (img_L - I_min) * ((1) / (I_max - I_min))
    
    return array_Global_histogram_stretching_L


##
## sceneRadiance.py
##
def sceneRadianceRGB(sceneRadiance):

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance


##
## Main
##
def process_file(input_path, output_path, file):
    try:
        filepath = input_path + "/" + file
        prefix = file.split('.')[0]
        
        if os.path.isfile(filepath):
            start_time = datetime.datetime.now()
            img = cv2.imread(input_path + '/' + file)
            sceneRadiance = RGB_equalisation(img)
            sceneRadiance = stretching(sceneRadiance)
            sceneRadiance = HSVStretching(sceneRadiance)
            sceneRadiance = sceneRadianceRGB(sceneRadiance)
            cv2.imwrite(output_path + '/' + prefix + '_UCM.jpg', sceneRadiance)
            end_time = datetime.datetime.now()
            print('******** File: ', file, ' Time: ', end_time - start_time, ' ********')
    except Exception as e:
        print(f"Error processing file: {file}")
        print(f"Error message: {str(e)}")
        print(traceback.format_exc())


if __name__ == '__main__':
    startTimeTotal = datetime.datetime.now()

    # Get dataset
    input_base_path = "./Dataset"

    # Process images from each folder
    for subfolder in os.listdir(input_base_path):
        if os.path.isdir(os.path.join(input_base_path, subfolder)):
            left_path = os.path.join(input_base_path, subfolder, "images/left")
            right_path = os.path.join(input_base_path, subfolder, "images/right")
            left_output_path = os.path.join(input_base_path, subfolder, "images_UCM/left")
            right_output_path = os.path.join(input_base_path, subfolder, "images_UCM/right")

            # Create output directories if they don't exist
            os.makedirs(left_output_path, exist_ok=True)
            os.makedirs(right_output_path, exist_ok=True)

            left_files = os.listdir(left_path)
            left_files = [file for file in left_files if file != '.DS_Store']  # Filter out '.DS_Store' files
            left_files = natsort.natsorted(left_files)

            right_files = os.listdir(right_path)
            right_files = [file for file in right_files if file != '.DS_Store']  # Filter out '.DS_Store' files
            right_files = natsort.natsorted(right_files)

            # Create a pool of worker processes
            num_processes = os.cpu_count()  # Use the number of CPU cores available
            pool = Pool(processes=num_processes)

            # Process left files in parallel
            pool.starmap(process_file, [(left_path, left_output_path, file) for file in left_files])

            # Process right files in parallel
            pool.starmap(process_file, [(right_path, right_output_path, file) for file in right_files])

            # Close the pool and wait for all processes to finish
            pool.close()
            pool.join()

    endTimeTotal = datetime.datetime.now()
    time = endTimeTotal - startTimeTotal
    print('time', time)