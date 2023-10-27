# blocks.py 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

# Applies a 2 convolutions + ReLu activation + batch normalization 
class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels: 
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True)  
        )
        
    def forward(self, dataIn): 
        return self.conv(dataIn)


class Down(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), 
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, dataIn): 
        return self.pool(dataIn)

class Up(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()    

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, dataIn, skipIn): 
        # skipIn: input from the skip connection (copy & crop)
        
        upsampled = self.up(dataIn)

        diffY = skipIn.size()[2] - upsampled.size()[2]
        diffX = skipIn.size()[3] - upsampled.size()[3]

        upsampled = F.pad(upsampled, [diffY//2, diffY-diffX // 2, diffY//2, diffY-diffX // 2])

        out = torch.cat([skipIn, upsampled], dim=1)
        return self.conv(out)

class Out(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, dataIn): 
        return self.conv(dataIn)
