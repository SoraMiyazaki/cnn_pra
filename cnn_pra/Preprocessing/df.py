import csv

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/mizutani/Desktop/code/data/poc/dataset.csv", header=None)
df = df.sample(n=30000, random_state=0)

np.savetxt("dataset.csv", df, fmt="%s")

data = pd.read_csv("dataset.csv", header=None).values.tolist()
df = []
for i in range(len(data)):
    a = "".join(data[i])
    b = a.split()
    df.append(b)

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(df)