"""
@Author: RyanHuang
@Data  : 20210910
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from onehot import getOneHot
from torchvision import transforms
import PIL
from PIL import Image


# pd.concat({pd.Series(pic6[i]) for i in range(len(pic6))},axis=0,ignore_index=True)
def get_vaild_list():
    pic6 = []
    eye2pic_pre = np.load('./data/npy/valid_eye2pic_pre.npy', allow_pickle=True).item()
    eye2pic = np.load('./data/npy/valid_eye2pic.npy', allow_pickle=True).item()
    for item in eye2pic_pre:
        if len(eye2pic_pre[item]) == 6:
            pic6.append({'ID': item, 'tag': 'pre', 'imglists': eye2pic_pre[item]})
    for item in eye2pic:
        if len(eye2pic[item]) == 6:
            pic6.append({'ID': item, 'tag': 'after', 'imglists': eye2pic[item]})

    return pic6


# 将数据集 4 : 1 分开
# trainSet、validSet 和 TestSet
# 筛选数据，只考虑6张图的情况
# train_eye2pic_pre = np.load('./data/npy/train_eye2pic_pre.npy', allow_pickle=True).item()
# train_eye2pic = np.load('./data/npy/train_eye2pic.npy', allow_pickle=True).item()
# valid_eye2pic_pre = np.load('./data/npy/valid_eye2pic_pre.npy', allow_pickle=True).item()
# valid_eye2pic = np.load('./data/npy/valid_eye2pic.npy', allow_pickle=True).item()


class TianChiData(Dataset):
    # 所有照片的后缀 有的图片文件夹 tmd 只有
    # SUFFIX = ["_100{}.jpg".format(i) for i in range(6)] + ["_200{}.jpg".format(i) for i in range(6)]

    def __init__(self, mode="train"):
        '''
        @Param:
            img_root 是包含 `0000-0000` 、 `0000-0004L`、`0000-0005` 等文件夹的目录
        '''
        pic6 = np.load('./data/npy/data_map.csv.npy', allow_pickle=True)
        test_raitio = 0.1
        train_num = int(len(pic6) * (1 - test_raitio))
        self.mode = mode

        if self.mode == 'train':
            self.data_DF = pic6[:train_num]
        elif self.mode == 'test':
            self.data_DF = pic6[train_num:]
        elif self.mode == 'vaild':
            self.data_DF = get_vaild_list()

    def __getitem__(self, index):
        if self.mode == 'vaild':
            currrent_Series = self.data_DF[index]
            img_list = currrent_Series["imglists"]
            img_concat = self.__read_img(img_list)
            return img_concat, currrent_Series["ID"], currrent_Series["tag"]
        else:
            currrent_Series = self.data_DF[index]
            patient_ID = currrent_Series["ID"]
            img_list = currrent_Series["imglists"]
            label = currrent_Series['label']
            code = currrent_Series['code']
            # 读取所有图片并堆叠
            img_concat = self.__read_img(img_list)
            code = torch.tensor(code, dtype=torch.float32)
            return img_concat, label, code

    def __len__(self):
        return len(self.data_DF)

    def __read_img(self, img_12_list):
        '''
        根据 img_12_list 读取图片
        并作图片裁剪, 并堆叠
        '''
        img_list = []
        for img_path in img_12_list:
            img_current = cv2.imread(img_path)

            img_source = cv2.imread(img_path.split('_seg')[0] + '.jpg')
            img_source = img_source[0:496, -638:-126]
            # resize samples
            img_source = cv2.resize(img_source, (256, 256), interpolation=cv2.INTER_NEAREST)

            img_current = img_current[:, :, 0].reshape(256, 256, 1)
            img_source = img_source[:, :, 0].reshape(256, 256, 1)
            img_list.append(img_source)
            img_list.append(img_current)

        img_array = np.concatenate(img_list, axis=-1)
        # 变成224*224
        # 变成224*224
        offset = np.random.randint(0, 32)
        img_array = img_array[offset:offset + 224, offset:offset + 224]
        return img_array.transpose(2, 0, 1).astype(np.float32)

    def __change_name(self, mistake_file_path):
        # 一个几乎用不到的函数...
        wrong_file_path, wrong_file_name = os.path.split(mistake_file_path)
        all_data_dir, wrong_file_parent = os.path.split(wrong_file_path)

        if "R" in wrong_file_name:
            wrong_file_name = wrong_file_name.replace("R", "L")
            # wrong_file_parent += "L"
        else:
            wrong_file_name = wrong_file_name.replace("L", "R")
            # wrong_file_parent += "R"

        return os.path.join(all_data_dir, wrong_file_parent, wrong_file_name)



