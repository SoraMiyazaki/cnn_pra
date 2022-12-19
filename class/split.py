import cv2
import re
import glob
from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

hd_data = sorted(glob.glob(r"C:\Users\mizutani\Desktop\CNN_pra\wave_image\hd\case01/*.jpg"), key=natural_keys)
hs_data = sorted(glob.glob(r"C:\Users\mizutani\Desktop\CNN_pra\wave_image\hs\case01/*.jpg"), key=natural_keys)
ud_data = sorted(glob.glob(r"C:\Users\mizutani\Desktop\CNN_pra\wave_image\ud\case01/*.jpg"), key=natural_keys)
us_data = sorted(glob.glob(r"C:\Users\mizutani\Desktop\CNN_pra\wave_image\us\case01/*.jpg"), key=natural_keys)

def data(data):
    arr = []
    for i in range(1, 1031, 21):
        arr.append(i)

    new_data = []
    for i in arr:
        new_data.append(data[i-1])
        
    return new_data

hd_data = data(hd_data)
hs_data = data(hs_data)
ud_data = data(ud_data)
us_data = data(us_data)

def resize(data):
    img = cv2.imread(data, 0)
    img = img[200:700, :]
    row, col = img.shape
    size = (col // 2, row // 2)
    img_resize = cv2.resize(img, size)

    img = img_resize[:, 0:500]
    # cv2.imwrite(out, img)
    
    return img

split_pixel = 50
split_img = 10

split_arr = []
def split(file, img):
    for i in range(0, 510, 10):
        split_arr.append(i)
    print(len(split_arr))
    exit()

    for i in range(split_img):
        for j in range(split_img):
            cv2.imwrite(f"data/poc/image/{file}_10/{file}{i+1}_{j+1}.jpg", img[:, split_arr[i]:split_arr[i+1]])
print(len(split_arr))
exit()
for file in tqdm(hd_data):
    split("hd", resize(file))

for file in tqdm(hs_data):
    split("hs", resize(file))    

for file in tqdm(ud_data):
    split("ud", resize(file))

for file in tqdm(us_data):
    split("us", resize(file))


