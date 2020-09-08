import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool

label_dic = np.load('label_dir.npy', allow_pickle=True).item()
print(label_dic)
testlist_name = "configs/testlist01.txt"

source_dir = 'data/UCF-101-jpg'
target_train_dir = 'data/UCF-101-jpg/train'
target_test_dir = 'data/UCF-101-jpg/test'
target_val_dir = 'data/UCF-101-jpg/val'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

for key in label_dic:
    print(key)

    label_dir = os.path.join(source_dir, key)
    label_mulu = os.listdir(label_dir)
    tag = 1
    label_mulu.sort()
    for each_label_mulu in label_mulu:
        image_file = os.listdir(os.path.join(label_dir, each_label_mulu))
        image_file.sort()
        frame = []

        for i in image_file:
            image_path = os.path.join(os.path.join(label_dir, each_label_mulu), i)
            frame.append(image_path)
        output_pkl = each_label_mulu + '.pkl'

        iftest = False
        for line in open(testlist_name, 'r'):
            if each_label_mulu in line[:-5]:
                iftest = True
                break

        if iftest:
            output_pkl = os.path.join(target_test_dir, output_pkl)
        else:
            output_pkl = os.path.join(target_train_dir, output_pkl)
        tag += 1
        f = open(output_pkl, 'wb')
        pickle.dump((each_label_mulu, label_dic[key], frame), f, -1)
        f.close()