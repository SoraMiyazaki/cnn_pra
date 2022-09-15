import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

files_h_d01 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case01/*.jpg"), reverse=False)
files_h_d02 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case02/*.jpg"), reverse=False)
files_h_d03 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case03/*.jpg"), reverse=False)
files_h_d04 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case04/*.jpg"), reverse=False)
files_h_d05 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case05/*.jpg"), reverse=False)
files_h_d06 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case06/*.jpg"), reverse=False)
files_h_d07 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case07/*.jpg"), reverse=False)
files_h_d08 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case08/*.jpg"), reverse=False)
files_h_d09 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case09/*.jpg"), reverse=False)
files_h_d10 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_遠目の画像/case10/*.jpg"), reverse=False)

files_h_s01 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case01/*.jpg"), reverse=False)
files_h_s02 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case02/*.jpg"), reverse=False)
files_h_s03 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case03/*.jpg"), reverse=False)
files_h_s04 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case04/*.jpg"), reverse=False)
files_h_s05 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case05/*.jpg"), reverse=False)
files_h_s06 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case06/*.jpg"), reverse=False)
files_h_s07 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case07/*.jpg"), reverse=False)
files_h_s08 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case08/*.jpg"), reverse=False)
files_h_s09 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case09/*.jpg"), reverse=False)
files_h_s10 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/上から_近目の画像/case10/*.jpg"), reverse=False)

files_u_d01 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case01/*.jpg"), reverse=False)
files_u_d02 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case02/*.jpg"), reverse=False)
files_u_d03 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case03/*.jpg"), reverse=False)
files_u_d04 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case04/*.jpg"), reverse=False)
files_u_d05 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case05/*.jpg"), reverse=False)
files_u_d06 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case06/*.jpg"), reverse=False)
files_u_d07 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case07/*.jpg"), reverse=False)
files_u_d08 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case08/*.jpg"), reverse=False)
files_u_d09 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case09/*.jpg"), reverse=False)
files_u_d10 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_遠目の画像/case10/*.jpg"), reverse=False)

files_u_s01 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case01/*.jpg"), reverse=False)
files_u_s02 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case02/*.jpg"), reverse=False)
files_u_s03 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case03/*.jpg"), reverse=False)
files_u_s04 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case04/*.jpg"), reverse=False)
files_u_s05 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case05/*.jpg"), reverse=False)
files_u_s06 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case06/*.jpg"), reverse=False)
files_u_s07 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case07/*.jpg"), reverse=False)
files_u_s08 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case08/*.jpg"), reverse=False)
files_u_s09 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case09/*.jpg"), reverse=False)
files_u_s10 = sorted(glob.glob("/Volumes/Untitled/CNN_pra/水平_近目の画像/case10/*.jpg"), reverse=False)

def resize(file, out_image):
    img = cv2.imread(file, 0)
    img = img[200:700, :] 

    # リサイズ
    row, col = img.shape
    size = (col // 2, row // 2)
    img_resize = cv2.resize(img, size)

    cv2.imwrite(out_image, img_resize)

for i in range(len(files_h_d01)):
    resize(files_h_d01[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case01{i+1}.jpg")
for i in range(len(files_h_d02)):
    resize(files_h_d02[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case02{i+1}.jpg") 
for i in range(len(files_h_d03)):
    resize(files_h_d03[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case03{i+1}.jpg")
for i in range(len(files_h_d04)):
    resize(files_h_d04[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case04{i+1}.jpg")           
for i in range(len(files_h_d05)):
    resize(files_h_d05[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case05{i+1}.jpg")
for i in range(len(files_h_d06)):
    resize(files_h_d06[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case06{i+1}.jpg")
for i in range(len(files_h_d07)):
    resize(files_h_d07[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case07{i+1}.jpg")
for i in range(len(files_h_d08)):
    resize(files_h_d08[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case08{i+1}.jpg")
for i in range(len(files_h_d09)):
    resize(files_h_d09[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case09{i+1}.jpg")
for i in range(len(files_h_d10)):
    resize(files_h_d10[i], f"/Volumes/Untitled/CNN_pra/resize_h_d/case10{i+1}.jpg")

for i in range(len(files_h_s01)):
    resize(files_h_s01[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case01{i+1}.jpg")
for i in range(len(files_h_s02)):
    resize(files_h_s02[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case02{i+1}.jpg") 
for i in range(len(files_h_s03)):
    resize(files_h_s03[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case03{i+1}.jpg")
for i in range(len(files_h_s04)):
    resize(files_h_s04[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case04{i+1}.jpg")           
for i in range(len(files_h_s05)):
    resize(files_h_s05[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case05{i+1}.jpg")
for i in range(len(files_h_s06)):
    resize(files_h_s06[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case06{i+1}.jpg")
for i in range(len(files_h_s07)):
    resize(files_h_s07[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case07{i+1}.jpg")
for i in range(len(files_h_s08)):
    resize(files_h_s08[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case08{i+1}.jpg")
for i in range(len(files_h_s09)):
    resize(files_h_s09[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case09{i+1}.jpg")
for i in range(len(files_h_s10)):
    resize(files_h_s10[i], f"/Volumes/Untitled/CNN_pra/resize_h_s/case10{i+1}.jpg")

for i in range(len(files_u_d01)):
    resize(files_u_d01[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case01{i+1}.jpg")
for i in range(len(files_u_d02)):
    resize(files_u_d02[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case02{i+1}.jpg") 
for i in range(len(files_u_d03)):
    resize(files_u_d03[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case03{i+1}.jpg")
for i in range(len(files_u_d04)):
    resize(files_u_d04[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case04{i+1}.jpg")           
for i in range(len(files_u_d05)):
    resize(files_u_d05[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case05{i+1}.jpg")
for i in range(len(files_u_d06)):
    resize(files_u_d06[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case06{i+1}.jpg")
for i in range(len(files_u_d07)):
    resize(files_u_d07[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case07{i+1}.jpg")
for i in range(len(files_u_d08)):
    resize(files_u_d08[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case08{i+1}.jpg")
for i in range(len(files_u_d09)):
    resize(files_u_d09[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case09{i+1}.jpg")
for i in range(len(files_u_d10)):
    resize(files_u_d10[i], f"/Volumes/Untitled/CNN_pra/resize_u_d/case10{i+1}.jpg")

for i in range(len(files_u_s01)):
    resize(files_u_s01[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case01{i+1}.jpg")
for i in range(len(files_u_s02)):
    resize(files_u_s02[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case02{i+1}.jpg") 
for i in range(len(files_u_s03)):
    resize(files_u_s03[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case03{i+1}.jpg")
for i in range(len(files_u_s04)):
    resize(files_u_s04[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case04{i+1}.jpg")           
for i in range(len(files_u_s05)):
    resize(files_u_s05[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case05{i+1}.jpg")
for i in range(len(files_u_s06)):
    resize(files_u_s06[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case06{i+1}.jpg")
for i in range(len(files_u_s07)):
    resize(files_u_s07[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case07{i+1}.jpg")
for i in range(len(files_u_s08)):
    resize(files_u_s08[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case08{i+1}.jpg")
for i in range(len(files_u_s09)):
    resize(files_u_s09[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case09{i+1}.jpg")
for i in range(len(files_u_s10)):
    resize(files_u_s10[i], f"/Volumes/Untitled/CNN_pra/resize_u_s/case10{i+1}.jpg")