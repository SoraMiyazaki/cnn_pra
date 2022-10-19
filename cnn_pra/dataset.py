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
  
data_hd = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\truth\\division_hd\\*.jpg"), key=natural_keys)
data_hs = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\truth\\division_hs\\*.jpg"), key=natural_keys)
data_ud = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\truth\\division_ud\\*.jpg"), key=natural_keys)
data_us = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\truth\\division_us\\*.jpg"), key=natural_keys)

for i in range(len(data_hd)):
    a = data_hd[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    data_g.append(a)

for i in range(len(data_hs)):
    a = data_hs[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    data_g.append(a)  

for i in range(len(data_ud)):
    a = data_ud[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    data_g.append(a)

for i in range(len(data_us)):
    a = data_us[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    data_g.append(a)     


file_w = []

file_hd = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\feature\\division_hd\\*.jpg"), key=natural_keys)
file_hs = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\feature\\division_hs\\*.jpg"), key=natural_keys)
file_ud = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\feature\\division_ud\\*.jpg"), key=natural_keys)
file_us = sorted(glob.glob("F:\\CNN_pra\\data\\poc\\image\\feature\\division_us\\*.jpg"), key=natural_keys)

for i in range(len(file_hd)):
    a = file_hd[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    file_w.append(a)

for i in range(len(file_hs)):
    a = file_hs[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    file_w.append(a)  

for i in range(len(file_ud)):
    a = file_ud[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    file_w.append(a)

for i in range(len(file_us)):
    a = file_us[i].replace("F:\\CNN_pra\\data\\poc\\", "")
    print(a)
    file_w.append(a) 



truth = np.array([data_g]).T
feature = np.array([file_w]).T

ds = np.append(feature, truth, axis=1)




np.savetxt("dataset.csv", ds, fmt="%s")


data = pd.read_csv("dataset.csv", header=None).values.tolist()
df = []
for i in range(len(data)):
    a = "".join(data[i])
    b = a.split()
    df.append(b)

with open("F:\\CNN_pra\\data\\poc\\dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(df)