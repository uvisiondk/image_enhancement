import os
import numpy as np
import cv2
import natsort
import xlwt
import datetime

from color_equalisation import RGB_equalisation
from global_histogram_stretching import stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# Get dataset
folder = "/home/aw/Code/UVision/image_processing/image_enhancement/Dataset"
path = folder + "/input"
files = os.listdir(path)
files = [file for file in files if file != '.DS_Store'] # Filter out '.DS_Store' files
files =  natsort.natsorted(files)


startTimeTotal = datetime.datetime.now()
for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        start_time = datetime.datetime.now()
        img = cv2.imread(folder + '/input/' + file)
        sceneRadiance = RGB_equalisation(img)
        sceneRadiance = stretching(sceneRadiance)
        sceneRadiance = HSVStretching(sceneRadiance)
        sceneRadiance = sceneRadianceRGB(sceneRadiance)
        cv2.imwrite('OutputImages/' + prefix + '_UCM.jpg', sceneRadiance)
        end_time = datetime.datetime.now()
        print('********    File: ',file, ' Time: ', end_time - start_time, '    ********')

endTimeTotal = datetime.datetime.now()
time = endTimeTotal-startTimeTotal
print('time',time)
