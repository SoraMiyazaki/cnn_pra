import cv2
import re
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# hd_data = sorted(glob.glob(r"waveimage/hd/*.jpg"), key=natural_keys)
# hs_data = sorted(glob.glob(r"waveimage/hs/*.jpg"), key=natural_keys)
# ud_data = sorted(glob.glob(r"waveimage/ud/*.jpg"), key=natural_keys)
# us_data = sorted(glob.glob(r"waveimage/us/*.jpg"), key=natural_keys)

# hd_data = hd_data[1000:1300]
# hs_data = hs_data[1000:1300]
# ud_data = ud_data[1000:1300]
# us_data = us_data[1000:1300]

# def resize(file, out):
#     img = cv2.imread(file, 0)
#     img = img[200:700, :]
#     row, col = img.shape
#     size = (col // 2, row // 2)
#     img_resize = cv2.resize(img, size)

#     img = img_resize[:, 0:500]
#     cv2.imwrite(out, img)
    
#     return img

# for i, name in enumerate(hd_data):
#     resize(name, f"resize_test/hd/resize{i+1}.jpg")

# for i, name in enumerate(hs_data):
#     resize(name, f"resize_test/hs/resize{i+1}.jpg")
    
# for i, name in enumerate(ud_data):
#     resize(name, f"resize_test/ud/resize{i+1}.jpg")
    
# for i, name in enumerate(us_data):
#     resize(name, f"resize_test/us/resize{i+1}.jpg")

hd = sorted(glob.glob(r"resize_test/hd/*.jpg"), key=natural_keys)
hs = sorted(glob.glob(r"resize_test/hs/*.jpg"), key=natural_keys)
ud = sorted(glob.glob(r"resize_test/ud/*.jpg"), key=natural_keys)
us = sorted(glob.glob(r"resize_test/us/*.jpg"), key=natural_keys)

split_arr = []
for i in range(0, 510, 10):
        split_arr.append(i)

for i in range(len(hd)):
    img = cv2.imread(hd[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/test50/hd/hd{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])    

for i in range(len(hs)):
    img = cv2.imread(hs[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/test50/hs/hs{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])   

for i in range(len(ud)):
    img = cv2.imread(ud[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/test50/ud/ud{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])   

for i in range(len(us)):
    img = cv2.imread(us[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/test50/us/us{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])   