import glob
import cv2
import numpy as np
import os

pic_paths = glob.glob(r'D:\project\Anti- VEGF\DME_WTF\data\*\*\*\*.png')
train_eye2pic_pre = {}
train_eye2pic = {}
valid_eye2pic_pre = {}
valid_eye2pic = {}
for pic_path in pic_paths:
    filename = pic_path.split('\\')[-1]
    eye_ID = filename.split('_')[0]
    tag = filename.split('_')[1][0]
    is_train = pic_path.split('\\')[-4] == 'train'
    if is_train:
        # 治疗前
        if tag == '1':
            if train_eye2pic_pre.__contains__(eye_ID):
                train_eye2pic_pre[eye_ID].append(pic_path)
            else:
                train_eye2pic_pre[eye_ID] = []
                train_eye2pic_pre[eye_ID].append(pic_path)
        else:  # 治疗后
            if train_eye2pic.__contains__(eye_ID):
                train_eye2pic[eye_ID].append(pic_path)
            else:
                train_eye2pic[eye_ID] = []
                train_eye2pic[eye_ID].append(pic_path)
    else:
        # 是验证集
        # 治疗前
        if tag == '1':
            if valid_eye2pic_pre.__contains__(eye_ID):
                valid_eye2pic_pre[eye_ID].append(pic_path)
            else:
                valid_eye2pic_pre[eye_ID] = []
                valid_eye2pic_pre[eye_ID].append(pic_path)
        else:  # 治疗后
            if valid_eye2pic.__contains__(eye_ID):
                valid_eye2pic[eye_ID].append(pic_path)
            else:
                valid_eye2pic[eye_ID] = []
                valid_eye2pic[eye_ID].append(pic_path)

np.save('./data/npy/train_eye2pic_pre.npy', train_eye2pic_pre)
np.save('./data/npy/train_eye2pic.npy', train_eye2pic)
np.save('./data/npy/valid_eye2pic_pre.npy', valid_eye2pic_pre)
np.save('./data/npy/valid_eye2pic.npy', valid_eye2pic)
