import re
import numpy as np
import glob
from PIL import Image
import time
import cv2

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

data_hd = sorted(glob.glob("/Volumes/Untitled/CNN_pra/01/train_hd/*.jpg"), key=natural_keys)
data_hs = sorted(glob.glob("/Volumes/Untitled/CNN_pra/01/train_hs/*.jpg"), key=natural_keys)
data_ud = sorted(glob.glob("/Volumes/Untitled/CNN_pra/01/train_ud/*.jpg"), key=natural_keys)
data_us = sorted(glob.glob("/Volumes/Untitled/CNN_pra/01/train_us/*.jpg"), key=natural_keys)



def div(data, count):
    img = np.array(Image.open(data).convert('L'))

    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_1.jpg", img[:, 0:100])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_2.jpg", img[:, 50:150])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_3.jpg", img[:, 100:200])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_4.jpg", img[:, 150:250])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_5.jpg", img[:, 200:300])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_6.jpg", img[:, 250:350])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_7.jpg", img[:, 300:400])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_8.jpg", img[:, 350:450])
    cv2.imwrite(f"/Volumes/Untitled/CNN_pra/data/poc/image/truth/thd_{count}_9.jpg", img[:, 400:500])

        
time_sta = time.time()
for i in range(len(data_hd)):
    div(data_hd[i], i+1) 
    print(i+1)   
time_end = time.time()
print((time_end - time_sta) / 60, "m")  