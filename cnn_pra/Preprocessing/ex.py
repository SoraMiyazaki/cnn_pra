import glob
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


file_hd = sorted(glob.glob("C:\\Users\\mizutani\\Desktop\\code\\data\\poc\\image\\feature\\division_hd\\*.jpg"), key=natural_keys)

print(file_hd[0])
