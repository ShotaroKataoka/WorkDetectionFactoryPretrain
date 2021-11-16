import os

import tensorboardX as tbx
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders.segment_dataloader import Segment_dataloader

from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.metrics import Evaluator
from modeling.cnn_lstm import CNN_LSTM
import yaml
# from model.mobilenet_LSTM import CNN_LSTM

def predict(date,loader):
    model.eval()
    evaluator.reset()
    tbar = tqdm(loader)
    test_loss = 0.0
    result = []
    for i, sample in enumerate(tbar):
        image, target, time = sample['image'], sample['label'], sample['time']
        if cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)

        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        test_loss += loss.item()
        
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis = 2)
        target = target.cpu().numpy()
        for batch, (row, trow) in enumerate(zip(pred, target)):
            for index, (ans, t) in enumerate(zip(row, trow)):
                result.append([time[batch][index], ans, t])


    result = np.array(result)
    np.savetxt('./result/result_' + str(date) + '.csv',result,delimiter=',')



"""
    load model
"""
with open("./params/predict_class.yaml", "r+") as f:
    params = yaml.safe_load(f)

cuda = True
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model = CNN_LSTM(100,256, 64, 12)
model = model.to(device)
weight_path = params['model']['weight_path']
checkpoint = torch.load(weight_path)
model.load_state_dict(checkpoint)

"""
    define parameter
"""
batch_size = 4
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = SegmentationLosses(cuda=cuda).build_loss(mode="ce")
evaluator = Evaluator(12)


"""
    load dataloader and prediction
"""
trainset = Segment_dataloader(mode="train")
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
predict('0903', train_loader)
del trainset
del train_loader


valset = Segment_dataloader(mode="val")
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
predict('0904', val_loader)
del valset
del val_loader

testset = Segment_dataloader(mode="test")
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
scheduler = LR_Scheduler("poly", 0.001, 100, len(test_loader))
predict('0915', test_loader)

exp = 1

print("start learning.")

