
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dual3DCNN(nn.Module):
    def __init__(self, width = 128, height = 128, depth = 128):
        super(Dual3DCNN, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        nr_filter = 32
        
        self.input_fixed = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True)            
        )
        
        self.input_moving = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True)            
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=nr_filter*2, out_features=64),
            # nn.Linear(in_features=nr_filter*2*(width//8)*(height//8)*(depth//8), out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    
    def forward(self, x_fixed, x_moving):
        out_fixed = self.input_fixed(x_fixed)
        out_moving = self.input_moving(x_moving)
        out = torch.cat((self.global_pool(out_fixed), self.global_pool(out_moving)), dim=1)
        out = out.view(out.size(0), -1)  # flatten the tensor for the Linear layer
        out = self.fc(out)
        return out
    
##############################################################################################################

class Dual3DCNN2(nn.Module):
    def __init__(self, width = 128, height = 128, depth = 128):
        super(Dual3DCNN2, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        nr_filter = 8
        
        self.input_fixed = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*2, out_channels=nr_filter*2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*2, out_channels=nr_filter*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*4, out_channels=nr_filter*4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*4, out_channels=nr_filter*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*8, out_channels=nr_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*8, out_channels=nr_filter*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*16, out_channels=nr_filter*16, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*16, out_channels=nr_filter*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True)
        )
        
        self.input_moving = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=nr_filter, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter, out_channels=nr_filter, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*2, out_channels=nr_filter*2, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*2, out_channels=nr_filter*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*4, out_channels=nr_filter*4, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*4, out_channels=nr_filter*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*8, out_channels=nr_filter*8, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*8, out_channels=nr_filter*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels=nr_filter*16, out_channels=nr_filter*16, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=nr_filter*16, out_channels=nr_filter*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=nr_filter),
            nn.ReLU(inplace=True)        
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=nr_filter*16, out_features=64),
            # nn.Linear(in_features=nr_filter*2*(width//8)*(height//8)*(depth//8), out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    
    def forward(self, x_fixed, x_moving):
        out_fixed = self.input_fixed(x_fixed)
        out_moving = self.input_moving(x_moving)
        out = torch.cat((self.global_pool(out_fixed), self.global_pool(out_moving)), dim=1)
        out = out.view(out.size(0), -1)  # flatten the tensor for the Linear layer
        out = self.fc(out)
        return out

##############################################################################################################


class Dual3DCNN3(nn.Module): #FC 128-64-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN3, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        num_filters = [8, 16, 32, 64, 128]
        num_blocks = len(num_filters)
        
        self.input_fixed_blocks = nn.ModuleList()
        self.input_moving_blocks = nn.ModuleList()
        
        for i in range(num_blocks):
            in_channels = 1 if i == 0 else num_filters[i-1]
            out_channels = num_filters[i]
            self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
            self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_filters[-1]*2, out_features=64),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def _make_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.LeakyReLU(inplace=True)
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

##############################################################################################################


class Dual3DCNN4(nn.Module): # FC 256-128-64-32-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN4, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        num_filters = [8, 16, 32, 64, 128]
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

##############################################################################################################

class Dual3DCNN5(nn.Module): # FC 256-128-64-32-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN5, self).__init__()
        
        self.initializer = nn.init.kaiming_normal_
        num_filters = [64, 128, 256]
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

##############################################################################################################

class Dual3DCNN6(nn.Module): # FC 256-128-64-32-3
    def __init__(self, width=128, height=128, depth=128):
        super(Dual3DCNN6, self).__init__()
        
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




##############################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

class Sama3DCNN4(nn.Module):
    def __init__(self, width=128, height=128, depth=128):
        super(Sama3DCNN4, self).__init__()

        # Hyperparameters
        self.num_filters = [8, 16, 32, 64, 128]
        self.output_scale = 15  # Scale factor for final output to map it to [-15, 15]

        # Initialize the convolutional blocks for both fixed and moving inputs
        self.input_fixed_blocks = nn.ModuleList()
        self.input_moving_blocks = nn.ModuleList()

        # Building convolutional blocks
        for i in range(len(self.num_filters)):
            in_channels = 1 if i == 0 else self.num_filters[i-1]
            out_channels = self.num_filters[i]
            self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
            self.input_moving_blocks.append(self._make_block(in_channels, out_channels))

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_filters[-1]*2, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3),
            nn.Tanh()  # Ensuring output is between -1 and 1
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x_fixed, x_moving):
        # Process each input through its respective blocks
        for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
            x_fixed = block_fixed(x_fixed)
            x_moving = block_moving(x_moving)

        # Pool and flatten the outputs
        x_fixed = self.global_pool(x_fixed).view(x_fixed.size(0), -1)
        x_moving = self.global_pool(x_moving).view(x_moving.size(0), -1)

        # Concatenate and pass through the fully connected layers
        x = torch.cat((x_fixed, x_moving), dim=1)
        x = self.fc(x)

        # Scale output to the range [-15, 15]
        return x * self.output_scale

######



import torch
import torch.nn as nn
import torch.nn.functional as F

# class Dual3DCNN4(nn.Module): # FC 256-128-64-32-3
#     def __init__(self, width=128, height=128, depth=128):
#         super(Dual3DCNN4, self).__init__()
        
#         self.initializer = nn.init.kaiming_normal_
#         num_filters = [16, 32, 64, 128]
#         num_blocks = len(num_filters)
        
#         self.input_fixed_blocks = nn.ModuleList()
#         self.input_moving_blocks = nn.ModuleList()
        
#         for i in range(num_blocks):
#             in_channels = 1 if i == 0 else num_filters[i-1]
#             out_channels = num_filters[i]
#             self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
#             self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=num_filters[-1]*2, out_features=128),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=128, out_features=64),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=64, out_features=32),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=32, out_features=3)
#         )
        
#         self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
#     def _make_block(self, in_channels, out_channels):
#         block = nn.Sequential(
#             nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=out_channels),
#             nn.LeakyReLU(inplace=True),
#         )
#         return block
    
#     def forward(self, x_fixed, x_moving):
#         out_fixed = x_fixed
#         out_moving = x_moving
#         for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
#             out_fixed = block_fixed(out_fixed)
#             out_moving = block_moving(out_moving)
        
#         out_fixed = self.global_pool(out_fixed)
#         out_moving = self.global_pool(out_moving)
        
#         out_fixed = out_fixed.view(out_fixed.size(0), -1)
#         out_moving = out_moving.view(out_moving.size(0), -1)
        
#         out = torch.cat((out_fixed, out_moving), dim=1)
#         out = self.fc(out)

#         return out
    



# class Dual3DCNN5(nn.Module): # FC 256-128-64-32-3
#     def __init__(self, width=128, height=128, depth=128):
#         super(Dual3DCNN5, self).__init__()
        
#         self.initializer = nn.init.kaiming_normal_
#         num_filters = [16, 32, 64, 128, 256]
#         num_blocks = len(num_filters)
        
#         self.input_fixed_blocks = nn.ModuleList()
#         self.input_moving_blocks = nn.ModuleList()
        
#         for i in range(num_blocks):
#             in_channels = 1 if i == 0 else num_filters[i-1]
#             out_channels = num_filters[i]
#             self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
#             self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=num_filters[-1]*2, out_features=128),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=128, out_features=64),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=64, out_features=32),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=32, out_features=3)
#         )
        
#         self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
#     def _make_block(self, in_channels, out_channels):
#         block = nn.Sequential(
#             nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=out_channels),
#             nn.LeakyReLU(inplace=True),
#         )
#         return block
    
#     def forward(self, x_fixed, x_moving):
#         out_fixed = x_fixed
#         out_moving = x_moving
#         for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
#             out_fixed = block_fixed(out_fixed)
#             out_moving = block_moving(out_moving)
        
#         out_fixed = self.global_pool(out_fixed)
#         out_moving = self.global_pool(out_moving)
        
#         out_fixed = out_fixed.view(out_fixed.size(0), -1)
#         out_moving = out_moving.view(out_moving.size(0), -1)
        
#         out = torch.cat((out_fixed, out_moving), dim=1)
#         out = self.fc(out)

#         return out
    


# class Dual3DCNN3(nn.Module): # FC 256-128-64-32-3
#     def __init__(self, width=128, height=128, depth=128):
#         super(Dual3DCNN3, self).__init__()
        
#         self.initializer = nn.init.kaiming_normal_
#         num_filters = [16, 32, 64]
#         num_blocks = len(num_filters)
        
#         self.input_fixed_blocks = nn.ModuleList()
#         self.input_moving_blocks = nn.ModuleList()
        
#         for i in range(num_blocks):
#             in_channels = 1 if i == 0 else num_filters[i-1]
#             out_channels = num_filters[i]
#             self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
#             self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
        
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=num_filters[-1]*2, out_features=128),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=128, out_features=64),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=64, out_features=32),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=32, out_features=3)
#         )
        
#         self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
#     def _make_block(self, in_channels, out_channels):
#         block = nn.Sequential(
#             nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=out_channels),
#             nn.LeakyReLU(inplace=True),
#         )
#         return block
    
#     def forward(self, x_fixed, x_moving):
#         out_fixed = x_fixed
#         out_moving = x_moving
#         for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
#             out_fixed = block_fixed(out_fixed)
#             out_moving = block_moving(out_moving)
        
#         out_fixed = self.global_pool(out_fixed)
#         out_moving = self.global_pool(out_moving)
        
#         out_fixed = out_fixed.view(out_fixed.size(0), -1)
#         out_moving = out_moving.view(out_moving.size(0), -1)
        
#         out = torch.cat((out_fixed, out_moving), dim=1)
#         out = self.fc(out)

#         return out



#########################################################################
# class FlexibleDual3DCNN(nn.Module):
#     def __init__(self, out_channels_list, in_channels=1):
#         super(FlexibleDual3DCNN, self).__init__()
        
#         self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
#         self.initializer = nn.init.kaiming_normal_
        
#         self.input_fixed_blocks = nn.ModuleList()
#         self.input_moving_blocks = nn.ModuleList()
        
#         # Adjust the loop to use out_channels_list directly
#         for out_channels in out_channels_list:
#             self.input_fixed_blocks.append(self._make_block(in_channels, out_channels))
#             self.input_moving_blocks.append(self._make_block(in_channels, out_channels))
#             in_channels = out_channels  # Update in_channels for the next iteration
        
#         # Assuming the last out_channels is the input feature size for the fully connected layer
#         self.fc = nn.Sequential(
#             nn.Linear(in_features=out_channels * 2, out_features=128),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=128, out_features=64),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=64, out_features=32),
#             nn.Dropout(p=0.5),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_features=32, out_features=3)
#         )
    
#     def _make_block(self, in_channels, out_channels):
#         block = nn.Sequential(
#             nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(num_features=out_channels),
#             nn.LeakyReLU(inplace=True),
#         )
#         return block
    
#     def forward(self, x_fixed, x_moving):
#         out_fixed = x_fixed
#         out_moving = x_moving
#         for block_fixed, block_moving in zip(self.input_fixed_blocks, self.input_moving_blocks):
#             out_fixed = block_fixed(out_fixed)
#             out_moving = block_moving(out_moving)
        
#         out_fixed = self.global_pool(out_fixed)
#         out_moving = self.global_pool(out_moving)
        
#         out_fixed = out_fixed.view(out_fixed.size(0), -1)
#         out_moving = out_moving.view(out_moving.size(0), -1)
        
#         out = torch.cat((out_fixed, out_moving), dim=1)
#         out = self.fc(out)

#         return out
