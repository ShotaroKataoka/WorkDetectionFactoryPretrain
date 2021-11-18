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


writer = tbx.SummaryWriter("log-1")
batch_size = 4
trainset = Segment_dataloader(mode="train")
valset = Segment_dataloader(mode="val")
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)


cuda = True
if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
model = CNN_LSTM(100,256, 64, 12)

model = model.to(device)
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
criterion = SegmentationLosses(cuda=cuda).build_loss(mode="ce")
evaluator = Evaluator(12)
scheduler = LR_Scheduler("poly", 0.001, 100, len(train_loader))

def training(epoch, best_pred):

    train_loss = 0.0
    model.train()
    num_img_tr = len(train_loader)
    tbar = tqdm(train_loader)
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if cuda:
            image, target = image.cuda(), target.cuda()
        scheduler(optimizer, i, epoch, best_pred)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        pred = output.data.cpu().numpy()
        batch = pred.shape[0]
        reshape_pred = np.zeros((batch,200),dtype=int)
        for index in range(batch):
            result =  np.argmax(pred[index], axis = 1)
            reshape_pred[index] =result
        # Add batch sample into evaluator
        target = target.cpu().numpy()
        evaluator.add_batch(target, reshape_pred)
        tbar.set_description('Train loss: %.7f' % (train_loss / (i + 1)))

    
    writer.add_scalar("train/loss", train_loss /(i + 1) , epoch) 
    Acc = evaluator.Pixel_Accuracy()
    writer.add_scalar("train/acc", Acc, epoch)

def validation(epoch, best_pred, best_loss):
    model.eval()
    evaluator.reset()
    tbar = tqdm(val_loader)
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = model(image)

        loss = criterion(output, target)
        test_loss += loss.item()
        
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        # Batches are set one at a time to argmax because 3D arrays cannot be converted to 2D arrays with argmax.
        
        batch = pred.shape[0]
        reshape_pred = np.zeros((batch,200),dtype=int)
        for index in range(batch):
            reshape_pred[index] = np.argmax(pred[index], axis = 1)
        # Add batch sample into evaluator
        evaluator.add_batch(target, reshape_pred)

        tbar.set_description('Test loss: %.7f' % (test_loss / (i + 1)))
    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    writer.add_scalar("Validation/loss", test_loss/(i + 1), epoch)
    writer.add_scalar("Validation/acc", Acc, epoch)
    writer.add_scalar("Validation/Acc_class", Acc_class, epoch)
    writer.add_scalar("Validation/mIoU", mIoU, epoch)
    writer.add_scalar("Validation/fwIoU", FWIoU, epoch)

    if test_loss/(i+1) < best_loss:
        best_loss = test_loss/(i+1)
    torch.save(model.state_dict(), "./model_weight/cnn_lstm/weight_epoc{}.pth".format(epoch + 1))
    print(f"Save ./model_weight/cnn_lstm/weight_epoc{epoch + 1}.pth")
    
    return best_pred, best_loss

exp = 1

print("start learning.")
best_pred = 0
best_loss = np.inf
t_losses = []
v_losses = []

for epoch in range(100):
    t_loss = training(epoch, best_pred)
    best_pred, best_loss = validation(epoch, best_pred, best_loss)




writer.close()
