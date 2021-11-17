"""
@Author: RyanHuang
@Data  : 20210911
"""
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from dataset import TianChiData
from model import TianChiModel
from onehot import getValue

MODEL_SAVE_PATH = "./output/best_model.pth"

# --------- 定义读取数据集的对象 ---------
vaild_Dataset = TianChiData(mode="vaild")

vaildloader = torch.utils.data.DataLoader(vaild_Dataset,
                                         shuffle=False,
                                         batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TianChiModel('resnet50', num_classes=159).to(device)
model.eval()
if True:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
result = []
res_csv = pd.read_csv('./data/submit.csv')
res_csv.set_index('patient ID', inplace=True)

with torch.no_grad():
    for (features, ID, tag) in tqdm(vaildloader):
        features = features.to(device)
        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        pred = getValue(predicted_labels)
        ID = ''.join(ID)
        tag = ''.join(tag)
        if str(tag) == 'pre':
            res_csv.loc[ID, 'preCST'] = pred
        else:
            res_csv.loc[ID, 'CST'] = pred

res_csv.to_csv('./data/result.csv')
