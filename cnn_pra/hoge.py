import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import re
import glob

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

img_path = sorted(glob.glob("../crush/resize_data/case01/*.jpg"), key=natural_keys)

# np.savetxt("img.csv", im, delimiter=',')

# hoge = pd.read_csv("img.csv", header=None).values
replace = pd.read_csv("replace.csv", header=None).values

def flat(diff2_min): 
        # 差分、平均を取得
        diff = np.append(0, np.diff(diff2_min)) 
        diff_avg = np.mean(np.abs(diff))
        
        # 差分が平均より大きい時に前後の平均を置換する
        diff_replacement = []
        for i in range(len(diff)):
            if np.abs(diff[i]) > diff_avg:
                if i == 0:
                    diff_replacement.append(diff2_min[1])
                elif i == len(diff)-1:
                    diff_replacement.append(diff2_min[-2])
                else:
                    diff_replacement.append((diff2_min[i-1] + diff2_min[i+1]) / 2)
                    diff[i+1] = diff2_min[i+1] - diff_replacement[i] 
            else:
                diff_replacement.append(diff2_min[i])

        return diff_replacement  

def draw_line(path, out_image, out_csv):
    im = cv2.imread(path, 0)

    for row in range(im.shape[1]):
        for col in range(im.shape[0]):
            if replace[col, row] == 255:
                im[col, row] = 200
    # np.savetxt("new_img.csv", im, delimiter=",")

    white = np.full([2, 500], 255)
    img = np.vstack((white, im))
    # print(img.shape)
    # exit()
    np.savetxt("new_img.csv", im, delimiter=",")

    cv2.imwrite("img.jpg", img)

    # cv2.imshow("img", img)

    # cv2.waitKey()

    img = cv2.imread("img.jpg", 0)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    dy = 1
    diff2_min = []

    for i in range(img.shape[1]):
        pixel = img[:, i]

        # 一階微分
        diff1 = [(int(pixel[j+1]) - int(pixel[j])) / dy for j in range(len(pixel)-1)]

        # 二階微分
        diff2 = [(int(diff1[k-1]) - int([n*2 for n in diff1][k]) + int(diff1[k+1])) / dy ** 2 for k in range(1, len(diff1)-1)]

        # 二階微分後の最小値のy座標を取得
        diff2_min.append(diff2.index(min(diff2)))  

        res = diff2_min            
        for _ in range(2000):
            res = flat(res) 

        res_ma = np.convolve(res, np.ones(13), 'valid') / 13
        a = np.array(res[-13:-1])
        res_ma = np.append(res_ma, a)

    x_lim = []
    for i in range(len(diff2_min)):
        x_lim.append(i+1)   


    np.savetxt(out_csv, res_ma, delimiter=",")

    fig = plt.figure(figsize=(5, 2.5))
    ax = fig.add_subplot(111)
    ax.plot(x_lim, res_ma, "r")
    ax.invert_yaxis()
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    image = Image.open(path)
    ax.imshow(image, cmap = "gray")
    # ax.invert_yaxis()
    plt.savefig(out_image)
    ax.cla()
    fig.clf()
    plt.close()

if __name__ == "__main__":
    for i, name in tqdm(enumerate(img_path)):
        draw_line(name, f"../crush/draw_data/case01/draw{i+1}.jpg", f"../crush/height_data/case01/height{i+1}.csv")
