# -*- coding: utf-8 -*-
# @Time    : 2020/12/24 3:57
# @Author  : youngleesin
# @FileName: process.py
# @Software: PyCharm

import numpy as np
from PIL import Image
import cv2
import os
from PIL import ImageEnhance
import math
import shutil
import matplotlib.pyplot as plt


# read the file
def unpickle(file):
    import _pickle as cPickle
    with open('./cifar-10-batches-py/'+file,'rb') as fo:
        dict = cPickle.load(fo,encoding='iso-8859-1')
    return dict

cnt_bird = 1
cnt_horse = 1
for j in range(5):
    mydata = unpickle('data_batch_{}'.format(j+1))
    X = mydata['data']
    label = mydata['labels']
    print(len(label))
    X = np.array(X)

    new = X.reshape(10000,3,32,32)

    for i in range(len(label)):
        if label[i] == 2:
            red   = new[i][0].reshape(1024,1)
            green = new[i][1].reshape(1024,1)
            blue  = new[i][2].reshape(1024,1)

            pic = np.hstack((red,green,blue))
            pic_rgb = pic.reshape(32,32,3)
            cv2.imwrite('bird/{}.jpg'.format(cnt_bird),pic_rgb)
            cnt_bird+=1

        elif label[i] == 7:
            red = new[i][0].reshape(1024, 1)
            green = new[i][1].reshape(1024, 1)
            blue = new[i][2].reshape(1024, 1)

            pic = np.hstack((red, green, blue))
            pic_rgb = pic.reshape(32, 32, 3)
            cv2.imwrite('horse/{}.jpg'.format(cnt_horse), pic_rgb)
            cnt_horse+=1


# data augment

def coppy():
    path = 'processed/bird_raw'
    paths = os.listdir(path)
    paths.sort(key=lambda x: int(x[:-4]))
    print(paths)
    cnt = len(paths)+1
    while True:
        if 200 <= len(os.listdir(path)):
            break
        for i in range(len(paths)):
            shutil.copyfile(path + '/' + paths[i], path + '/{}.jpg'.format(cnt))
            cnt += 1

    path = 'processed/horse_raw'
    paths = os.listdir(path)
    paths.sort(key=lambda x: int(x[:-4]))
    print(paths)
    cnt = len(paths) + 1
    while True:
        if 200 <= len(os.listdir(path)):
            break
        for i in range(len(paths)):
            shutil.copyfile(path + '/' + paths[i], path + '/{}.jpg'.format(cnt))
            cnt += 1

def mirror():
    path = 'processed/bird_raw'
    paths = os.listdir(path)
    print(paths)
    i = 201
    for pt in paths:
        mydata = cv2.imread(path + "/{}".format(pt))
        # print(image.shape, type(image))
        img1 = np.fliplr(mydata)
        # print(img1)
        cv2.imwrite('processed/bird_raw/{}.jpg'.format(i), img1)
        i += 1

    path = 'processed/horse_raw'
    paths = os.listdir(path)
    print(paths)
    i = 201
    for pt in paths:
        mydata = cv2.imread(path + "/{}".format(pt))
        # print(image.shape, type(image))
        img1 = np.fliplr(mydata)
        # print(img1)
        cv2.imwrite('processed/horse_raw/{}.jpg'.format(i), img1)
        i += 1

def fun_Contrast(image, coefficient, path_save):
    # 对比度，增强因子为1.0是原始图片
    # increase 1.5
    # decrease 0.8
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted1 = enh_con.enhance(coefficient)
    image_contrasted1.save(path_save)

def my_aug():
    file_root = "processed/bird_raw/"
    save_root = "processed/bird_raw/"
    paths = os.listdir(file_root)
    cnt = len(paths)+1
    for pt in paths:
        path = file_root + pt
        image = Image.open(path)
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.2, path_save_contra)
        cnt += 1
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.3, path_save_contra)
        cnt += 1
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.4, path_save_contra)
        cnt += 1
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.5, path_save_contra)
        cnt += 1

    file_root = "processed/horse_raw/"
    save_root = "processed/horse_raw/"
    paths = os.listdir(file_root)
    cnt = len(paths)+1
    for pt in paths:
        path = file_root + pt
        image = Image.open(path)
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.2, path_save_contra)
        cnt += 1
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.3, path_save_contra)
        cnt += 1
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.4, path_save_contra)
        cnt += 1
        path_save_contra = save_root + "{}.jpg".format(cnt)
        fun_Contrast(image, 1.5, path_save_contra)
        cnt += 1


if __name__ == "__main__":
    my_aug()
    coppy()
    mirror()
