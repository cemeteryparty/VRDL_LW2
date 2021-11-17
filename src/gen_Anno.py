from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat
import os, sys, csv, cv2
import pandas as pd
import numpy as np

DigitStruct = loadmat("digitStruct.mat")
mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
img_path_list = []
x1_list = []
y1_list = []
x2_list = []
y2_list = []
category_list = []
for idx in range(33402):
    fname = "{}.png".format(idx + 1)
    im = cv2.imread("train/" + fname)
    bboxes = DigitStruct[fname].astype(np.int32)
    for i in range(bboxes.shape[0]):
        img_path_list.append("train/" + fname)
        x1_list.append(max(bboxes[i][1], 0))
        y1_list.append(max(bboxes[i][2], 0))
        x2_list.append(min(bboxes[i][1] + bboxes[i][3], im.shape[1] - 1))
        y2_list.append(min(bboxes[i][2] + bboxes[i][0], im.shape[0] - 1))
        category_list.append(mapper[int(bboxes[i][4])])
    # assert os.path.exists("train/" + fname)
    # assert fname in DigitStruct.keys()
anno = pd.DataFrame()
anno["img_path"] = img_path_list
anno["x1"] = x1_list
anno["y1"] = y1_list
anno["x2"] = x2_list
anno["y2"] = y2_list
anno["class"] = category_list
tra_anno, val_anno = train_test_split(anno, test_size=0.25, random_state=1116)
tra_anno.to_csv("tra_annotations.csv", index=None, header=None)
val_anno.to_csv("val_annotations.csv", index=None, header=None)

with open("classes.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(10):
        writer.writerow([str(i), i])
print(len(tra_anno), len(val_anno))
