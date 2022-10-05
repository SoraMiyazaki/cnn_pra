import numpy as np
import re
import glob
import cv2

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

data_g = []    
  
data_hd = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_hd/*.jpg"), key=natural_keys)
data_hs = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_hs/*.jpg"), key=natural_keys)
data_ud = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_ud/*.jpg"), key=natural_keys)
data_us = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/truth/division_us/*.jpg"), key=natural_keys)

for i in range(len(data_hd)):
    data_g.append(data_hd[i])

for i in range(len(data_hs)):
    data_g.append(data_hs[i])  

for i in range(len(data_ud)):
    data_g.append(data_ud[i])

for i in range(len(data_us)):
    data_g.append(data_us[i])          

file_w = []

file_hd = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/feature/division_hd/*.jpg"), key=natural_keys)
file_hs = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/feature/division_hs/*.jpg"), key=natural_keys)
file_ud = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/feature/division_ud/*.jpg"), key=natural_keys)
file_us = sorted(glob.glob("/Volumes/Untitled/CNN_pra/data/poc/image/feature/division_us/*.jpg"), key=natural_keys)

for i in range(len(file_hd)):
    file_w.append(file_hd[i])

for i in range(len(file_hs)):
    file_w.append(file_hs[i])  

for i in range(len(file_ud)):
    file_w.append(file_ud[i])

for i in range(len(file_us)):
    file_w.append(file_us[i]) 

truth = np.array([data_g]).T
feature = np.array([file_w]).T

ds = np.append(feature, truth, axis=1).tolist()


np.savetxt("/Volumes/Untitled/CNN_pra/data/PoC/dataset.csv", ds, fmt="%s")

# a = np.loadtxt("/Volumes/Untitled/CNN_pra/data/PoC/dataset.csv", dtype="str")

# def write(img, out_img):
#     image = cv2.imread(img)
#     cv2.imwrite(out_img, image)
    
# for i in range(len(truth)):
#     print(i+1)
#     write(data_g[i], f"/Volumes/Untitled/CNN_pra/data/PoC/image/truth/truth{i+1}.jpg")
    

# for i in range(len(feature)):
#     print(i+1)
#     write(file_w[i], f"/Volumes/Untitled/CNN_pra/data/PoC/image/feature/feature{i+1}.jpg")    

