import torch
import torch.nn as nn
import torch.nn.functional as F

class Dual3DCNN4(nn.Module): # FC 256-128-64-32-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN4, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        num_filters = [16, 32, 64, 128]
        num_blocks = len(num_filters)
        
        self.input_fixed_blocks = nn.ModuleList()
        self.input_moving_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            in_channels = 1 if i == 0 else num_filters[i-1]
            out_channels = num_filters[i]
            self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
            self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_filters[-1]*2, out_features=128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _make_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return block
    
    def forward(self, x_fixed, x_moving):
        out_fixed = x_fixed
        out_moving = x_moving
        for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
            out_fixed = block_fixed(out_fixed)
            out_moving = block_moving(out_moving)
        
        out_fixed = self.global_pool(out_fixed)
        out_moving = self.global_pool(out_moving)
        
        out_fixed = out_fixed.view(out_fixed.size(0), -1)
        out_moving = out_moving.view(out_moving.size(0), -1)
        
        out = torch.cat((out_fixed, out_moving), dim=1)
        out = self.fc(out)

        return out
    



class Dual3DCNN5(nn.Module): # FC 256-128-64-32-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN5, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        num_filters = [16, 32, 64, 128, 256]
        num_blocks = len(num_filters)
        
        self.input_fixed_blocks = nn.ModuleList()
        self.input_moving_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            in_channels = 1 if i == 0 else num_filters[i-1]
            out_channels = num_filters[i]
            self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
            self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_filters[-1]*2, out_features=128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _make_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return block
    
    def forward(self, x_fixed, x_moving):
        out_fixed = x_fixed
        out_moving = x_moving
        for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
            out_fixed = block_fixed(out_fixed)
            out_moving = block_moving(out_moving)
        
        out_fixed = self.global_pool(out_fixed)
        out_moving = self.global_pool(out_moving)
        
        out_fixed = out_fixed.view(out_fixed.size(0), -1)
        out_moving = out_moving.view(out_moving.size(0), -1)
        
        out = torch.cat((out_fixed, out_moving), dim=1)
        out = self.fc(out)

        return out
    


class Dual3DCNN3(nn.Module): # FC 256-128-64-32-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN3, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        num_filters = [16, 32, 64]
        num_blocks = len(num_filters)
        
        self.input_fixed_blocks = nn.ModuleList()
        self.input_moving_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            in_channels = 1 if i == 0 else num_filters[i-1]
            out_channels = num_filters[i]
            self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
            self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_filters[-1]*2, out_features=128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _make_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
        )
        return block
    
    def forward(self, x_fixed, x_moving):
        out_fixed = x_fixed
        out_moving = x_moving
        for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
            out_fixed = block_fixed(out_fixed)
            out_moving = block_moving(out_moving)
        
        out_fixed = self.global_pool(out_fixed)
        out_moving = self.global_pool(out_moving)
        
        out_fixed = out_fixed.view(out_fixed.size(0), -1)
        out_moving = out_moving.view(out_moving.size(0), -1)
        
        out = torch.cat((out_fixed, out_moving), dim=1)
        out = self.fc(out)

        return out
