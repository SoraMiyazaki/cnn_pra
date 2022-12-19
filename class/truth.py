import glob
import re
import statistics
import numpy as np
import pandas as pd

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# dataset
data_hd = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_h_d/*.csv"), key=natural_keys)
data_hs = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_h_s/*.csv"), key=natural_keys)
data_ud = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_u_d/*.csv"), key=natural_keys)
data_us = sorted(glob.glob("C:/Users/mizutani/Desktop/CNN_pra/height/height_u_s/*.csv"), key=natural_keys)

data_hd = data_hd[0:1057]
data_hs = data_hs[0:1057]
data_ud = data_ud[0:1057]
data_us = data_us[0:1057]

def data(data):
    arr = []
    for i in range(1, 1031, 21):
        arr.append(i)
    
    new_data = []
    for i in arr:
        new_data.append(data[i-1])
        
    return new_data

split_pixel = 50
split_img = 10

waterline_height_avg = []
split_arr = []
def draw(data):
    for i in range(0, 510, split_pixel):
        split_arr.append(i)
        
    # dataset    
    a = pd.read_csv(data, header=None).values
    # test
    # a = data
    a1 = np.round(a, decimals=0)

    img = np.zeros((250, 512))

    for i in range(img.shape[1]): #img.shape[1]
        for j in range(img.shape[0]):
            if j == int(a1[i]):
                img[j-1, i] = 1   

    waterline_height = []
    
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            if img[j, i] == 1:
                waterline_height.append(j+1)
                
    for j in range(split_img):
        waterline_height_avg.append(int(statistics.mean(waterline_height[split_arr[j]:split_arr[j+1]])))
        # cv2.imwrite(f"test_img/resize/split/{file}{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]]) 

for file in data(data_hd):
    draw(file)

for file in data(data_hs):
    draw(file)

for file in data(data_ud):
    draw(file)

for file in data(data_us):
    draw(file)

# dataset
file_hd = sorted(glob.glob("data/poc/image/hd_10/*.jpg"), key=natural_keys)
file_hs = sorted(glob.glob("data/poc/image/hs_10/*.jpg"), key=natural_keys)
file_ud = sorted(glob.glob("data/poc/image/ud_10/*.jpg"), key=natural_keys)
file_us = sorted(glob.glob("data/poc/image/us_10/*.jpg"), key=natural_keys)

print(len(file_hd))
exit()

feature_list = []

for i in range(len(file_hd)):
    a = file_hd[i].replace("data\\poc\\", "")
    feature_list.append(a)

for i in range(len(file_hs)):
    a = file_hs[i].replace("data\\poc\\", "")
    feature_list.append(a)  

for i in range(len(file_ud)):
    a = file_ud[i].replace("data\\poc\\", "")
    feature_list.append(a)

for i in range(len(file_us)):
    a = file_us[i].replace("data\\poc\\", "")
    feature_list.append(a) 

truth = np.array([waterline_height_avg]).T
feature = np.array([feature_list]).T
print(truth.shape, feature.shape)
exit()


ds = np.append(feature, truth, axis=1)

np.savetxt("dataset.csv", ds, delimiter=',', fmt="%s")