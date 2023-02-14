import numpy as np
import re
import glob
import cv2
import pandas as pd
import csv

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

data_g = []    
  
data_hd = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\hd\\truth\\*.jpg"), key=natural_keys)
data_hs = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\hs\\truth\\*.jpg"), key=natural_keys)
data_ud = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\ud\\truth\\*.jpg"), key=natural_keys)
data_us = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\us\\truth\\*.jpg"), key=natural_keys)

for i in range(len(data_hd)):
    data_hd[i] = f"image\\test\\hd\\truth\\thd_10570_{i+1}.jpg"
    data_g.append(data_hd[i])

for i in range(len(data_hs)):
    data_hs[i] = f"image\\test\\hs\\truth\\ths_10570_{i+1}.jpg"
    data_g.append(data_hs[i])  

for i in range(len(data_ud)):
    data_ud[i] = f"image\\test\\ud\\truth\\tud_10570_{i+1}.jpg"
    data_g.append(data_ud[i])

for i in range(len(data_us)):
    data_us[i] = f"image\\test\\us\\truth\\tus_10570_{i+1}.jpg"
    data_g.append(data_us[i])     


file_w = []

file_hd = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\hd\\feature\\*.jpg"), key=natural_keys)
file_hs = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\hs\\feature\\*.jpg"), key=natural_keys)
file_ud = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\ud\\feature\\*.jpg"), key=natural_keys)
file_us = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\test\\us\\feature\\*.jpg"), key=natural_keys)

for i in range(len(file_hd)):
    file_hd[i] = f"image\\test\\hd\\feature\\dhd_10570_{i+1}.jpg"
    file_w.append(file_hd[i])

for i in range(len(file_hs)):
    file_hs[i] = f"image\\test\\hs\\feature\\dhs_10570_{i+1}.jpg"
    file_w.append(file_hs[i])  

for i in range(len(file_ud)):
    file_ud[i] = f"image\\test\\ud\\feature\\dud_10570_{i+1}.jpg"
    file_w.append(file_ud[i])

for i in range(len(file_us)):
    file_us[i] = f"image\\test\\us\\feature\\dus_10570_{i+1}.jpg"
    file_w.append(file_us[i]) 

truth = np.array([data_g]).T
feature = np.array([file_w]).T

ds = np.append(feature, truth, axis=1)

np.savetxt("test.csv", ds, fmt="%s")

data = pd.read_csv("test.csv", header=None).values.tolist()
df = []
for i in range(len(data)):
    a = "".join(data[i])
    b = a.split()
    df.append(b)

with open("F:\\CNN_pra\\data\\poc\\test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(df)