import numpy as np
import pandas as pd
import glob
import re
import time
import cv2

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

data_hd = sorted(glob.glob("/Volumes/Untitled/CNN_pra/height/height_h_d/*.csv"), key=natural_keys)
data_hs = sorted(glob.glob("/Volumes/Untitled/CNN_pra/height/height_h_s/*.csv"), key=natural_keys)
data_ud = sorted(glob.glob("/Volumes/Untitled/CNN_pra/height/height_u_d/*.csv"), key=natural_keys)
data_us = sorted(glob.glob("/Volumes/Untitled/CNN_pra/height/height_u_s/*.csv"), key=natural_keys)

def draw(data, count):
    a = pd.read_csv(data, header=None).values
    a1 = np.round(a, decimals=0)

    img = np.zeros((250, 512))

    for i in range(img.shape[1]): #img.shape[1]
        for j in range(img.shape[0]):
            if j == int(a1[i]):
                img[j-1, i] = 1          

    # np.savetxt(out_img, img)
    # cv2.imwrite(out_img, img) 

    

    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_1.jpg", img[:, 0:100])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_2.jpg", img[:, 50:150])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_3.jpg", img[:, 100:200])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_4.jpg", img[:, 150:250])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_5.jpg", img[:, 200:300])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_6.jpg", img[:, 250:350])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_7.jpg", img[:, 300:400])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_8.jpg", img[:, 350:450])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/tus_{count}_9.jpg", img[:, 400:500])
 

    # cv2.imread("test.jpg", 0)       
    # print(type(img[:, 267]))  



# for i in range(len(data_hd)):
#     draw(data_hd[i], f"/Volumes/Untitled/CNN_pra/gray/gray_hd/ghd_{i+1}.jpg")

# for i in range(len(data_hs)):
#     draw(data_hs[i], f"/Volumes/Untitled/CNN_pra/gray/gray_hs/ghs_{i+1}.jpg")
   
# for i in range(len(data_ud)):
#     draw(data_ud[i], f"/Volumes/Untitled/CNN_pra/gray/gray_ud/gud_{i+1}.jpg")
  
# for i in range(len(data_us)):
#     draw(data_us[i], f"/Volumes/Untitled/CNN_pra/gray/gray_us/gus_{i+1}.jpg")           

for i in range(len(data_us)):
    draw(data_us[i], i+1)
    print(i+1)

# for i in range(len(data_hs)):
#     draw(data_hs[i], i+1)
   
# for i in range(len(data_ud)):
#     draw(data_ud[i], i+1)
  
# for i in range(len(data_us)):
#     draw(data_us[i], i+1)   