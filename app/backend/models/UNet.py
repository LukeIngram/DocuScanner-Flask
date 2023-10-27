# unet.py

import torch.nn as nn 

from .blocks import * 

class UNet(nn.Module): 
    def __init__(self, n_channels, n_classes, n_blocks=4, start=32): 
        super(UNet, self).__init__()
    
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.start = start
  
        self.layers = nn.Sequential(
            ConvBlock(n_channels, start), 
            *self.get_blocks(start), 
            Out(start, n_classes)
        )

    def forward(self, dataIn):
        num_layers = len(self.layers)

        outs = [dataIn] # Store outputs for skip connections
        for i in range(0, self.n_blocks+1):
            outs.append(self.layers[i].forward(outs[-1]))

        out = outs.pop()
        for i in range(self.n_blocks+1, num_layers-1):
            out = self.layers[i].forward(out, outs.pop())
        
        logits = self.layers[-1].forward(out)
        return logits


    # generates up/down blocks based on defined amount (hyperparameter)   
    def get_blocks(self, start): 
        blocks = []
        for i in range(self.n_blocks):
            start_mult = start * 2 ** i
            blocks.append(Down(start_mult, start_mult * 2))
        for i in range(self.n_blocks-1, -1, -1):
            start_mult = start * 2 ** i
            blocks.append(Up(start_mult * 2, start_mult))
        return blocks
    
    def __str__(self): 
        return 'UNet'
    
