import numpy as np
import glob
import cv2
import time
from multiprocessing import Pool
import argparse

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

    for i in range(img_resize.shape[1]):
        pixel = blur[:, i]

        # 一階微分
        diff1 = [(int(pixel[j+1]) - int(pixel[j])) / dy for j in range(len(pixel)-1)]

        # 二階微分
        diff2 = [(int(diff1[k-1]) - int([n*2 for n in diff1][k]) + int(diff1[k+1])) / dy ** 2 for k in range(1, len(diff1)-1)]

        # 二階微分後の最小値のy座標を取得
        diff2_min.append(diff2.index(min(diff2)))      

    res = diff2_min            
    for _ in range(2000):
        res = flat(res)

    # 移動平均
    res_ma = np.convolve(res, np.ones(13), 'valid') / 13
    a = np.array(res[-13:-1])
    res_ma = np.append(res_ma, a)

    y_array = []
    for i in range(len(diff2_min)):
        y_array.append(i+1)      

    np.savetxt(out_image, res_ma)

    print(f"exit: {out_image}")

def draw_line_batch(args):
    draw_line(*args)

def main(case, multi):
    time_sta = time.time()
    argslist = []
    # 複数画像の読み込み
    files_h_d = sorted(glob.glob(f"/Volumes/Untitled/CNN_pra/水平_遠目の画像/case{case:02}/*.jpg"), reverse=False)
    files_h_s = sorted(glob.glob(f"/Volumes/Untitled/CNN_pra/水平_近目の画像/case{case:02}/*.jpg"), reverse=False)
    files_u_d = sorted(glob.glob(f"/Volumes/Untitled/CNN_pra/上から_遠目の画像/case{case:02}/*.jpg"), reverse=False)
    files_u_s = sorted(glob.glob(f"/Volumes/Untitled/CNN_pra/上から_近目の画像/case{case:02}/*.jpg"), reverse=False)

    for i in range(len(files_h_d)):
        argslist.append([files_h_d[i], f"/Volumes/Untitled/CNN_pra/height_h_d/hhd{case:02}_{i+1}.csv"])
    for i in range(len(files_h_s)):
        argslist.append([files_h_s[i], f"/Volumes/Untitled/CNN_pra/height_h_s/hhs{case:02}_{i+1}.csv"])
    for i in range(len(files_u_d)):
        argslist.append([files_u_d[i], f"/Volumes/Untitled/CNN_pra/height_u_d/hud{case:02}_{i+1}.csv"])
    for i in range(len(files_u_s)):
        argslist.append([files_u_s[i], f"/Volumes/Untitled/CNN_pra/height_u_s/hus{case:02}_{i+1}.csv"])

    if multi:
        with Pool(4) as pp:
            pp.map(draw_line_batch, argslist)
    else:
        for args in argslist:
            draw_line(*args)
            
    time_end = time.time()
    Time = (time_end - time_sta) / 60
    print(Time, "m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-s", "--start", type=int, default=1)
    parser.add_argument("-c", "--case", type=int)
    parser.add_argument("--multi", action="store_true")
    args = parser.parse_args()

    if args.case is not None and 1<=args.case and args.case<=10:
        main(args.case, args.multi)
    else:
        for case in range(args.start, 11):
            main(case, args.multi)
            if case != 10:
                time.sleep(600)