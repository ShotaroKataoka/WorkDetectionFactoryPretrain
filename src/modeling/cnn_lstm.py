import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, stride=1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2 ,padding=14)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(256)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(256)

    def forward(self, x):
        """
        Parameters
        ----------
        x: todo
            torch.Size([1600, 256, 30, 40])
            4-D Tensor either of shape (b * sequence_length, c,h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        
        x = self.relu(self.norm1(self.conv1(x))) # in : torch.Size([1600, 256, h, w]) out :torch.Size([800, 256, h-2, w-2])
        x = self.relu(self.norm2(self.conv2(x))) # in : torch.Size([1600, 256, h-2, w-2]) out :torch.Size([800, 256, h-4, w-4])
        x = self.relu(self.norm3(self.conv3(x))) # in : torch.Size([1600, 256, h-4, w-4]) out :torch.Size([800, 256, ((h-4)/2 )-1, ((w-6)/2) -1])
        return x

class CnnModule(nn.Module):
    def __init__(self):
        super(CnnModule, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, stride=2) 
        self.conv2 = nn.Conv2d(256, 256, 3, stride=2)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(256)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(256)

    def forward(self, x):
        """
        Parameters
        ----------
        x: todo
            torch.Size([1600, 256, 30, 40])
            4-D Tensor either of shape (b * sequence_length, c,h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        
        x = self.relu(self.norm1(self.conv1(x))) # in : torch.Size([1600, 256, 30, 40]) out :torch.Size([800, 256, 28, 38])
        x = self.relu(self.norm2(self.conv2(x)))  # in : torch.Size([800, 256, 28, 38]) out :torch.Size([800, 256, 26, 36])
        x = self.relu(self.norm3(self.conv3(x)))   # in : torch.Size([800, 256, 26, 36]) out :torch.Size([800, 256, 12, 17])
        # ↓ Global average pooling
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]) # in: torch.Size([1600, 512, 28, 38]) out: torch.Size([1600, 512, 1, 1])

        sequence_length , c ,h,w = x.shape 
        x = x.view(sequence_length, c)  # in: torch.Size([1600, 512, 1, 1]) out: torch.Size([1600, 512])
        # ↑ Global average pooling
        return x

class CNN_LSTM(nn.Module):
    def __init__(self,sequence_length, input_dim, hidden_dim, nclass, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, nclass)
        self.block1 = Block()  
        self.block2 = Block()  
        self.cnn = CnnModule()  
        self.soft = nn.Softmax(dim = 1)
         
    def forward(self, x):
        """
        Parameters
        ----------
        x: todo
            torch.Size([batch, 200 , 256, 30, 40])
            5-D Tensor either of shape (b, sequence_length, c,h, w)

        Returns
        -------
        last_state_list, layer_output
        """
        x=x.float() 
        b, sequence_length,c, h ,w = x.shape
        x = x.view(b * sequence_length, c,h,w) # torch.Size([200 * batch,  3, 480, 640])
        cnn_result = self.cnn(x) #in :  torch.Size([200 * batch,  3, 480, 640]) out: torch.Size([1600, 128])
        cnn_result = cnn_result.view(b,sequence_length, -1) # in: torch.Size([1600, 128]) out:  shape torch.Size([8, 200, 128])
        lstm_out, (h_n, c_n) = self.lstm(cnn_result)  # in: shape torch.Size([8, 200, 128]) out: torch.Size([8, 200, 64])
        lstm_out = self.dropout(lstm_out)
        tag_score = self.hidden2tag(lstm_out) # in: torch.Size([8, 200, 64]) out: torch.Size([8, 200, 13])

        return tag_score
