import cv2
import re
import glob
from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# hd_data = sorted(glob.glob(r"waveimage/hd/*.jpg"), key=natural_keys)
# hs_data = sorted(glob.glob(r"waveimage/hs/*.jpg"), key=natural_keys)
# ud_data = sorted(glob.glob(r"waveimage/ud/*.jpg"), key=natural_keys)
# us_data = sorted(glob.glob(r"waveimage/us/*.jpg"), key=natural_keys)

# hd_data = hd_data[0:1000]
# hs_data = hs_data[0:1000]
# ud_data = ud_data[0:1000]
# us_data = us_data[0:1000]

# def resize(data, out):
#     img = cv2.imread(data, 0)
#     img = img[200:700, :]
#     row, col = img.shape
#     size = (col // 2, row // 2)
#     img_resize = cv2.resize(img, size)

#     img = img_resize[:, 0:500]
#     cv2.imwrite(out, img)
    
#     return img

# for i, name in enumerate(hd_data):
#     resize(name, f"resize_data/hd/resize{i+1}.jpg")

# for i, name in enumerate(hs_data):
#     resize(name, f"resize_data/hs/resize{i+1}.jpg")
    
# for i, name in enumerate(ud_data):
#     resize(name, f"resize_data/ud/resize{i+1}.jpg")
    
# for i, name in enumerate(us_data):
#     resize(name, f"resize_data/us/resize{i+1}.jpg")

hd = sorted(glob.glob(r"resize_data/hd/*.jpg"), key=natural_keys)
hs = sorted(glob.glob(r"resize_data/hs/*.jpg"), key=natural_keys)
ud = sorted(glob.glob(r"resize_data/ud/*.jpg"), key=natural_keys)
us = sorted(glob.glob(r"resize_data/us/*.jpg"), key=natural_keys)

split_arr = []
for i in range(0, 510, 10):
    split_arr.append(i)


for i in tqdm(range(len(hd))):
    img = cv2.imread(hd[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/train50/hd/hd{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])    

for i in tqdm(range(len(hs))):
    img = cv2.imread(hs[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/train50/hs/hs{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])   

for i in tqdm(range(len(hd))):
    img = cv2.imread(ud[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/train50/ud/ud{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])   

for i in tqdm(range(len(hd))):
    img = cv2.imread(us[i])
    
    for j in range(50):
        cv2.imwrite(f"data/poc/image/train50/us/us{i+1}_{j+1}.jpg", img[:, split_arr[j]:split_arr[j+1]])   


