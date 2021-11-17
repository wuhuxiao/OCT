import os
import time
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from dataset import TianChiData
from model import TianChiModel
import numpy as np
import pandas as pd
from tqdm import tqdm
from onehot import getValue
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = './output'
LOGFILE = os.path.join(PATH, 'training.log')
# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % device)
header.append('Output Path: %s' % PATH)

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()
# --------- some 超参数 ---------
learning_rate = 0.005
NUM_CLASSES = 159
loss_mse_w = 1

EPOCH = 200
MODEL_SAVE_PATH = os.path.join(PATH, 'best.pth')

# --------- 定义读取数据集的对象 ---------
BATCH_SIZE = 16
train_Dataset = TianChiData(mode="train")
trainloader = torch.utils.data.DataLoader(train_Dataset,
                                          shuffle=True,
                                          batch_size=BATCH_SIZE,
                                          drop_last=True)
test_Dataset = TianChiData(mode="test")

testloader = torch.utils.data.DataLoader(test_Dataset,
                                         shuffle=False,
                                         batch_size=1)

###########################################
# Initialize Cost, Model, and Optimizer
###########################################
imp = torch.ones(NUM_CLASSES - 1, dtype=torch.float)


def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels
                       + F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
    return torch.mean(val)


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


model = TianChiModel('resnet50', num_classes=159).to(device)
imp = imp.to(device)

if True and os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
loss_func_MSE = torch.nn.MSELoss()
loss_func_CrossEntropy = torch.nn.CrossEntropyLoss()

start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1
# --------- 模型初始化 ---------
total_loss = 0
try:
    for epoch in range(EPOCH):
        model.train()
        for batch_idx, (features, targets, code) in enumerate(trainloader):

            features = features.to(device)
            code = code.to(device)

            # FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = cost_fn(logits, code, imp)
            optimizer.zero_grad()
            # UPDATE MODEL PARAMETERS
            cost.backward()
            optimizer.step()

            # LOGGING
            if not batch_idx % 50:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                     % (epoch + 1, EPOCH, batch_idx,
                        len(train_Dataset) // BATCH_SIZE, cost))
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)
        scheduler.step()
        # 每个epoch测试一下效果，保留效果最后的
        model.eval()
        with torch.no_grad():
            valid_mae, valid_mse = compute_mae_and_mse(model, testloader,
                                                       device=device)
        if valid_mae < best_mae:
            best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
            ########## SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))

        s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
            valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time) / 60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)


except KeyboardInterrupt:
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("保存完毕!!")
