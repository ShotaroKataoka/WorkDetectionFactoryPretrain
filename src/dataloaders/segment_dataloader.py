import numpy as np
import torch
from torch.utils.data import Dataset
import os
import csv
from torchvision import transforms
import dataloaders.custom_transforms as tr
from PIL import Image, ImageFile
from glob import glob
from sklearn.utils import shuffle
import pandas as pd
import json
import re
from tqdm import tqdm
import psutil
import yaml
import datetime

class Segment_dataloader:
    def __init__(self, mode="train", length=400, rate=0.8):
        """
            Shape the data for learning and use it for learning.
            Store the formatted data in self.data
            self.data = {'03072732': {'label': '', 'start': start_image_path, index: 'start_path_index'}}
            Parameters
            ----------
            mode : string
                One of train, val, or test.
            length : integer
                data lenght
            rate : integer
                train,val rate
        """
        with open("./params/predict_class.yaml", "r+") as f:
          self.params = yaml.safe_load(f)
        self.mode = mode
        self.length = length
        self.endlength= 400//2
        self.y = []
        self.x = []
        self.data = []
        self.npz_list = []
        self.csv_list = []
        self.none_time = []
        self.timelist = []
        self.transform = transforms.Compose([transforms.ToTensor()])
        index = 0

        
        if mode=="train":
            base_path = self.params['dataloader']['base_path']
            train_npz_list = self.params['dataloader']['train_npz_path']
            train_csv_list = self.params['dataloader']['train_csv_list']

            # ------ load npz file --------------
            self.load_npz_csv(base_path, train_npz_list, train_csv_list)

            # ----------- create train data -------------
            for order, label_path in enumerate(self.params['dataloader']['train_label_list']):
                label = pd.read_csv(label_path).values
                time_dict = self.csv_list[order]
                if order == 0 :
                    for index in range(len(label)):
                        time = label[index][0].split(':')
                        self.set_data('03', time, time_dict, label, index)
                        if index % self.params['dataloader']['slice_num'] == 0 :
                            self.data.append(label[index][1])
                            

                else :
                    for index in range(len(label)):
                        time = label[index][0].split(':')
                        if int(time[0]) < self.params['dataloader']['04split'] :
                            self.set_data('04', time, time_dict, label, index)
                            if index %  self.params['dataloader']['slice_num'] == 0 :
                                self.data.append(label[index][1])

        elif mode=="val":
            base_path = self.params['dataloader']['base_path']
            val_npz_list = self.params['dataloader']['val_npz_path']
            val_csv_list = self.params['dataloader']['val_csv_list']

            # ------ load npz file --------------
            self.load_npz_csv(base_path, val_npz_list, val_csv_list)

            # ----------- create train data -------------

            for order, label_path in enumerate(self.params['dataloader']['val_label_list']):
                label = pd.read_csv(label_path).values
                time_dict = self.csv_list[order]

                for index in range(len(label)):
                    time = label[index][0].split(':')
                    if int(time[0]) > self.params['dataloader']['04split'] :
                        self.set_data('04', time, time_dict, label, index)
                        if index %  self.params['dataloader']['slice_num'] == 0 :
                            self.data.append(label[index][1])

        elif mode=="test":
            base_path = self.params['dataloader']['base_path']
            val_npz_list = self.params['dataloader']['test_npz_path']
            val_csv_list = self.params['dataloader']['test_csv_list']

            # ------ load npz file --------------
            self.load_npz_csv(base_path, val_npz_list, val_csv_list)

            # ----------- create train data -------------

            for order, label_path in enumerate(self.params['dataloader']['test_label_list']):
                label = pd.read_csv(label_path).values
                time_dict = self.csv_list[order]

                for index in range(len(label)):
                    time = label[index][0].split(':')
                    self.set_data('15', time, time_dict, label, index)
                    if index %  self.params['dataloader']['slice_num'] == 0 :
                        self.data.append(label[index][1])

        if mode == "test":
            self.data = self.data[0: -157]
        else:
            self.data = self.data[0: -45]

        
    def __getitem__(self, index):
        _img , _target , _time = self.load_image(index)
        # sample = {'image': self.transform(_img), 'label': self.transform(_target)}
        sample = {'image': torch.from_numpy(_img).float(), 'label': torch.from_numpy(_target).float(), 'time': _time}
        return sample
        
    def __len__(self):
        return len(self.data) - 40
    

    def load_image(self, index):
        """
            Parameters
            ----------
            index : int
                first index number 
        """
        # print(index, len(self.x), len(self.y))
        index = index * self.params['dataloader']['slice_num'] * 2
        index_list  = self.x[index]
        y_data = self.y[index : index + self.endlength]
        if index_list[0] == '03' and len(self.npz_list) == 2:
            x_data = self.npz_list[0][index_list[1] : index_list[1] + self.endlength]
        elif index_list[0] == '04' and len(self.npz_list) == 2:
            x_data = self.npz_list[1][index_list[1] : index_list[1] + self.endlength]
        else:
            x_data = self.npz_list[0][index_list[1] : index_list[1] + self.endlength]

        if len(x_data) != 200:
            print('enter in if under 200', len(x_data))
            x_data = x_data.to_list()
            while len(x_data) < 200:
                x_data.append(x_data[-1])
                y_data.append(y_data[-1])

        return x_data , np.array(y_data), np.array(self.timelist[index: index + self.endlength])
    

    def format_time(self, day, times):
        times = times.split(':')
        return day + str(times[0]).zfill(2) + str(times[1]).zfill(2) + str(times[2]).zfill(2)

    def get_image(self, time, day):
        image = []
        y = []
        if( len(self.data[time]['index']) == 2):
            for index in self.data[time]['index']:
                try:
                    if(day == '03'):
                        y.append(self.data[time]['y'] - 1)
                        image.append(self.npz_list[0][index])
                    elif (day == '04' and len(self.npz_list) != 1):
                        y.append(self.data[time]['y'] - 1)
                        image.append(self.npz_list[1][index])
                    elif (len(self.npz_list) == 1):
                        y.append(self.data[time]['y'] - 1)
                        image.append(self.npz_list[0][index])
                except Exception as e:
                    print('error get image in if ', e)

        else:
            try:
                index = self.data[time]['index']
                if(day == '03'):
                    y.append(self.data[time]['y'] -1)
                    y.append(self.data[time]['y'] -1)
                    image.append(self.npz_list[0][index][0])
                    image.append(self.npz_list[0][index][0])
                elif (day == '04' and len(self.npz_list) != 1):
                    y.append(self.data[time]['y'] -1)
                    y.append(self.data[time]['y'] -1)
                    image.append(self.npz_list[1][index][0])
                    image.append(self.npz_list[1][index][0])
                elif (len(self.npz_list) == 1):
                    y.append(self.data[time]['y'] -1)
                    y.append(self.data[time]['y'] -1)
                    image.append(self.npz_list[0][index][0])
                    image.append(self.npz_list[0][index][0])
            except Exception as e:
                print('error get image in else ', e)
        return image , np.array(y)

    def load_npz_csv(self,base_path, train_npz_list, train_csv_list):
        for order, path in enumerate(tqdm(train_npz_list)):
            self.npz_list.append( np.load(base_path + path + '.npz')['arr_0'] )

            csv_dict = pd.read_csv(base_path + train_csv_list[order] , index_col=0 ,names=('path','time')).to_dict(orient='index')
            new_dict = {}
            for index, k in enumerate(list(csv_dict)):
                key = str(csv_dict[k]['time'])
                new_dict.setdefault(key, {'path': [], 'index': []})
                new_dict[ key ]['path'].append(k)
                new_dict[ key ]['index'].append(index)
            self.csv_list.append(new_dict)

    def get_target(self, time, time_dict,date):
        target_time =  str(int(time[0]))+ str(time[1]).zfill(2) + str(time[2]).zfill(2)
        try:
            target = time_dict[target_time]
        except:
            target = None
            self.none_time.append(target_time)

        time = int(date) * 1000000 + int(time[0])* 10000 + int(time[1]) * 100 + int(time[2])
        return target, time

    def set_data(self, date, time, time_dict, label, index):
        target, time = self.get_target(time ,time_dict,date)
        label_dict = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 11:9, 12:10, 13:11}
        if target != None:
            if 2 == len(target['index']):
                for start in range(len(target['index'])):
                    self.timelist.append(time)
                    self.y.append(label_dict[label[index][1]])
                    self.x.append([date,target['index'][start]])
            else:
                self.y.append(label_dict[label[index][1]])
                self.y.append(label_dict[label[index][1]])
                self.x.append([date,target['index'][0]])
                self.x.append([date,target['index'][0]])

                self.timelist.append(time)
                self.timelist.append(time)

    def transform_data(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor()
        ])
        return composed_transforms(sample)  





