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

MODEL_SAVE_PATH = "./output/test_best.pth"

# --------- 定义读取数据集的对象 ---------
test_Dataset = TianChiData(mode="vaild")

testloader = torch.utils.data.DataLoader(test_Dataset,
                                         shuffle=False,
                                         batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TianChiModel('resnet50', input_channel=12, regr_out=1).to(device)
model.eval()
if True:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
result = []
res_csv = pd.read_csv('./data/submit.csv')
res_csv.set_index('patient ID', inplace=True)

with torch.no_grad():
    for (X, ID, tag) in tqdm(testloader):
        X = X.to(device)
        reg_output = model(X)
        pred = reg_output.cpu().numpy().argmax()
        pred = getValue(pred)
        ID = ''.join(ID)
        tag = ''.join(tag)
        result.append({'ID': ID, 'tag': tag, 'pred': pred})
        if str(tag) == 'pre':
            res_csv.loc[ID, 'preCST'] = pred
        else:
            res_csv.loc[ID, 'CST'] = pred
result = np.array(result)
np.save('./data/npy/result.npy', result)

res_csv.to_csv('./data/result.csv')
