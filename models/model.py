
#import torch
import torch.nn as nn


class AnimalClassifierNet(nn.Module):
    
    def __init__(self,num_classes=3,dropout=0.3):
        super().__init__()
        
        self.Block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(dropout)
        )
        
        self.Block2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(dropout)
        )
        
        self.Block3 = nn.Sequential(
            
            nn.Conv2d(128,256,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(dropout)
        )
        
        
        self.HeadBlock = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*16*16,512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512,num_classes)
        )
        
    
    
    def forward(self,x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.HeadBlock(x)
        return x
        
