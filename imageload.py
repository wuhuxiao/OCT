import glob
import cv2
import numpy as np
import os
import pandas as pd
from onehot import getOneHot
# 修改为相应的目录
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
# 均衡样本每类180
def get_list():
    pic6 = []
    # scale = 1500
    scale = 1
    # 0000-1717L,2,45,4,2,1,6109,1,1,0,0,2,1,303,1,1,0,1 去掉这条异常
    data_df = pd.read_csv('./data/TrainingAnnotation.csv').reset_index(drop=True)
    data_df.set_index(['patient ID'], inplace=True)
    label_num = pd.read_csv('./data/label_num.csv').reset_index(drop=True)
    label_num.set_index(['label'], inplace=True)
    train_eye2pic_pre = np.load('./data/npy/train_eye2pic_pre.npy', allow_pickle=True).item()
    train_eye2pic = np.load('./data/npy/train_eye2pic.npy', allow_pickle=True).item()
    for item in train_eye2pic_pre:
        if len(train_eye2pic_pre[item]) == 6 and data_df.loc[item, 'preCST'] > 0:
            CST = data_df.loc[item, 'preCST']
            label, code = getOneHot(CST.astype(np.float32))
            # 均衡样本
            for i in range(178 // label_num.loc[label, 'nums']):
                # 最多重复80次
                if i >= 80:
                    break
                pic6.append({'ID': item, 'label': label, 'code': code, 'imglists': train_eye2pic_pre[item]})
    for item in train_eye2pic:
        if len(train_eye2pic[item]) == 6 and data_df.loc[item, 'CST'] > 0:
            CST = data_df.loc[item, 'CST']
            label, code = getOneHot(CST.astype(np.float32))
            # 均衡样本
            for i in range(178 // label_num.loc[label, 'nums']):
                # 最多重复80次
                if i >= 80:
                    break
                pic6.append({'ID': item, 'label': label, 'code': code, 'imglists': train_eye2pic[item]})

    pic6 = np.array(pic6)
    np.random.shuffle(pic6)
    np.save('./data/npy/data_map.csv', pic6)
    df = pd.concat((pd.Series(pic6[i]) for i in range(len(pic6))), axis=1, ignore_index=True)
    df = df.T
    df = df.sample(frac=1.0)
    df.to_csv('./data/data_map.csv')
    return pic6

get_list()