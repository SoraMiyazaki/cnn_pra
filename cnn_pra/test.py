import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import time

# 複数画像の読み込み
files_u_d = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case04/*.jpg"), reverse=False)
files_u_s = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case04/*.jpg"), reverse=False)
# files_h_d = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case04/*.jpg"), reverse=False)
files_h_s = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case04/*.jpg"), reverse=False)

# 波線の調整
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

def draw_line(file, out_image):
    dy = 1
    diff2_min = []
    img = cv2.imread(file, 0)
    img = img[200:700, :] # 画像のトリミング(ndarray)

    # リサイズ
    row, col = img.shape
    size = (col // 2, row // 2)
    img_resize = cv2.resize(img, size)

    # ガウシアン
    blur = cv2.GaussianBlur(img_resize, (9, 9), 0)

    # キャニー
    edges = cv2.Canny(blur,100,200)

    cv2.imwrite("resize.jpg", img_resize)
    cv2.imwrite("blur.jpg", blur)
    cv2.imwrite("edges.jpg", edges)
    
        
    for i in range(img_resize.shape[1]):
        pixel = blur[:, i]

        # 一階微分
        diff1 = []
        for j in range(len(pixel)-1):
            first_diff = (int(pixel[j+1]) - int(pixel[j])) / dy
            diff1.append(first_diff)

        # 二階微分
        diff2 = []
        for k in range(1, len(diff1)-1):
            second_diff = (int(diff1[k-1]) - int([n*2 for n in diff1][k]) + int(diff1[k+1])) / dy ** 2
            diff2.append(second_diff)

        # 二階微分後の最小値のy座標を取得
        diff2_min.append(diff2.index(min(diff2)))      

    res = diff2_min            
    for _ in range(1500):
        res = flat(res)

    # 移動平均
    res_ma = np.convolve(res, np.ones(13), 'valid') / 13
    a = np.array(res[-13:-1])
    res_ma = np.append(res_ma, a)

    y_array = []
    for i in range(len(diff2_min)):
        y_array.append(i+1)      

    fig = plt.figure(figsize=(5.12, 2.5))
    ax = fig.add_subplot(1, 1, 1)
    im = cv2.imread("/Users/miyazakisora/Python/cnn_pra/resize.jpg", 0)
    ax.imshow(im, cmap="gray")
    ax.plot(y_array, res_ma, "r")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.xlim(0, 512)
    plt.ylim(225, 0)
    plt.savefig(out_image)
    plt.close()

# num = 0
# time_sta = time.time()
# for i in range(len(files_h_d)):
#     if i == 0:
#         print("水平・遠目")
#     num += 1
#     print(num, "枚目")
#     draw_line(files_h_d[i], "/Volumes/Untitled/CNN_pra/waveline_h_d/hd04_" + str(i+1)+ ".jpg")
# time_end = time.time()
# Time = (time_end - time_sta) / 60
# print(Time, "m")

num = 0
time_sta = time.time()
for i in range(len(files_h_s)):
    if i == 0:
        print("水平・近目")
    num += 1
    print(num, "枚目")
    draw_line(files_h_s[i], f"/Volumes/Untitled/CNN_pra/waveline_h_s/hs04_{i+1}.jpg")
time_end = time.time()
Time = (time_end - time_sta) / 60
print(Time, "m")

num = 0
time_sta = time.time()
for i in range(len(files_u_d)):
    if i == 0:
        print("上部・遠目")
    num += 1
    print(num, "枚目")
    draw_line(files_u_d[i], f"/Volumes/Untitled/CNN_pra/waveline_u_d/ud04_{i+1}.jpg")
time_end = time.time()
Time = (time_end - time_sta) / 60
print(Time, "m")

num = 0
time_sta = time.time()
for i in range(len(files_u_s)):
    if i == 0:
        print("上部・近目")
    num += 1
    print(num, "枚目")
    draw_line(files_u_s[i], f"/Volumes/Untitled/CNN_pra/waveline_u_s/us04_{i+1}.jpg")    
time_end = time.time()
Time = (time_end - time_sta) / 60
print(Time, "m")