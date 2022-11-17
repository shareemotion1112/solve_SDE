import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
            
        #image-tensor goes in as batch_sizex3x32x32
        #print-1 will show this state
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        #image-tensor is batch_sizex16x32x32 since: (32-3+2*1)/1+1=32
        #print-2 will show this state
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        #image-tensor is batch_sizex32x16x16 since: (32-3+2*1)/2+1=16
        #print-3 will show this state
        
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #image-tensor is batch_sizex32x8x8
        #print-4 will show this state

        #now we flatten image tensor to batch_sizex32*8*8 which is batch_sizex2048
        #print-5 will show this state
        self.fc1 = nn.Linear(32 * 8 * 8, 10) #same as: self.fc1 = nn.Linear(2048, 10)
            
            
    def forward(self, x):
        print("print-1:")
        print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        print("print-2:")
        print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        print("print-3:")
        print(x.shape)
        x = self.max_pool(x)
        print("print-4:")
        print(x.shape)
        x = x.view(-1, 32 * 8 * 8)
        print("print-5:")
        print(x.shape)
        x = self.fc1(x)
        print("fc1")
        print(x.shape)
        return x
    
model = CNN()

input = torch.randn(4, 3, 32, 32)
model(input)