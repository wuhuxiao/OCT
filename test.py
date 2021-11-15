"""
@Author: RyanHuang
@Data  : 20210911
"""
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataset import TianChiData
from model import TianChiModel
from tqdm import tqdm
from onehot import getValue
MODEL_SAVE_PATH = "./output/test_best.pth"

# --------- 定义读取数据集的对象 ---------
test_Dataset = TianChiData( mode="test")
# test_Dataset = TianChiData( mode="train")


testloader  = torch.utils.data.DataLoader(test_Dataset,
                                          shuffle=False,
                                          batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TianChiModel('resnet50', input_channel=12, regr_out=1).to(device)
if True:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
result = []
predctions = []
soruce = []
model.eval()
with torch.no_grad():
    for (X, y, ID) in tqdm(testloader):
        X = X.to(device)

        reg_output = model(X)
        pred = reg_output.cpu().numpy().argmax()
        pred = getValue(pred)
        y = y.numpy().max()
        y = getValue(y)
        gap = np.abs(y - pred)/y
        predctions.append(pred)
        soruce.append(y)
        if gap.max() < 0.025:
            result.append(1)
        else:
            result.append(0)

result = np.array(result)
for i in range(0,len(predctions)):
    print(soruce[i],predctions[i])
print('acc',result.sum()/len(result))
