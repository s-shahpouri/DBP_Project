import os
import numpy as np
import torch
from tqdm import tqdm
import json
from dual_network_3f import Dual3DCNN6 as Dual
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import glob
from utilities import create_list_from_master_json, read_json_file, split_data
import re
import glob
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import torch
from utilities import list_patient_folders, prepare_data_nrrd, split_data
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, ScaleIntensityd, SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import LoadImaged
from monai.data.image_reader import ITKReader
from monai.data import SmartCacheDataset

from utilities import list_patient_folders, prepare_data_nrrd, split_data

# Specify the directory where the patient folders are located
data_path_NEW = '/home/shahpouriz/Data/DBP_newDATA/DBP/nrrd/proton'
data_path_OLD = '/home/shahpouriz/Data/DBP_oldDATA/nrrd/proton'

# Get the list of patient folders

patient_list_NEW = list_patient_folders(data_path_NEW)
pct, rct, pos = prepare_data_nrrd(data_path_NEW, patient_list_NEW)
data_NEW = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct, rct, pos)]


patient_list_OLD = list_patient_folders(data_path_OLD)
pct, rct, pos = prepare_data_nrrd(data_path_OLD, patient_list_OLD)
data_OLD = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct, rct, pos)]


# Assuming data_NEW and data_OLD are your lists of dictionaries
data = data_NEW + data_OLD
# data = data_NEW[:20] + data_OLD[:20]
# data = data_NEW 

# Split the data
train_data, val_data, test_data = split_data(data)

# Check the lengths of the sets
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of test samples:", len(test_data))

# Set parameters
starting_epoch = 0
# decay_epoch = 20
final_epoch = 100
# learning_rate = 0.0001
lambda_reg = 0.000001

# Condition for saving list
# save_list = False
best_mae = np.inf

exception_list = ['']

#### My method

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import LoadImaged
from monai.data.image_reader import ITKReader
batch_size = 4
dim = 128
size = (dim, dim, dim)
transforms = Compose([
        LoadImaged(keys=["plan", "repeat"], reader=ITKReader()),
        EnsureChannelFirstd(keys=["plan", "repeat"]),
        ScaleIntensityd(keys=["plan", "repeat"]),
        Spacingd(keys=["plan", "repeat"], pixdim=(3.0, 3.0, 3.0), mode='trilinear'),
        SpatialPadd(keys=["plan", "repeat"], spatial_size=size, mode='constant'),  # Ensure minimum size
        CenterSpatialCropd(keys=["plan", "repeat"], roi_size=size),  # Ensure uniform size
    ])


# train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=1.0, num_workers=2)
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

# val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=1.0, num_workers=2)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

train_ds = SmartCacheDataset(data=train_data, transform=transforms, cache_rate=0.1, replace_rate=0.2, num_init_workers=2, num_replace_workers=2)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

val_ds = SmartCacheDataset(data=val_data, transform=transforms, cache_rate=0.1, replace_rate=0.2, num_init_workers=2, num_replace_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)


# Build model
print('Initializing model...')
model = Dual(width=dim, height=dim, depth=dim)
device = torch.device("cuda:0")
model.to(device)

# Define loss
print('Defining loss...')
mae_loss = torch.nn.L1Loss()
# mae_loss = torch.nn.MSELoss()

# Optimizer


# lr = 1e-5
# weight_decay = 1e-3
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epoch, eta_min=1e-7)


# from decayLR import DecayLR
# decay_epoch = 50
# learning_rate = 1e-3
# offset = 1e-6
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# lr_lambda = DecayLR(epochs=final_epoch, offset=offset, decay_epochs=decay_epoch).step
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


from torch.optim.lr_scheduler import ReduceLROnPlateau
learning_rate = 1e-4  # Start with a learning rate of 10^-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Scheduler (optional, to reduce learning rate when a metric has stopped improving)
# Here, we use ReduceLROnPlateau which reduces learning rate when a metric stops improving
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


# Initialize best_mae before the training loop
best_mae = float('inf')
train_losses = []

# Training loop
for epoch in range(starting_epoch, final_epoch):
    model.train()  # Set model to training mode
    mae_list = []
    train_loss = []
    # Assuming mae_loss is defined and using the correct device
    for i, batch_data in enumerate(train_loader):
        pCT, rCT = batch_data["plan"].to(device), batch_data["repeat"].to(device)
        reg = batch_data["pos"].to(device)  # No need for clone().detach().requires_grad_(True) for targets
        
        # Forward pass
        outx, outy, outz = model(pCT, rCT)
        
        # Split reg into its components (assuming reg has shape [batch_size, 3])
        regx, regy, regz = reg[:, 0], reg[:, 1], reg[:, 2]

        # Calculate MAE for each component
        loss_x = mae_loss(outx, regx.unsqueeze(1))  # Add dimension to match output shape
        loss_y = mae_loss(outy, regy.unsqueeze(1))
        loss_z = mae_loss(outz, regz.unsqueeze(1))

        # Combine the losses (you could also weigh them differently)
        total_loss = (loss_x + loss_y + loss_z)/batch_size
        train_losses.append(total_loss.item())
        
        # Calculate average validation loss
        avg_train_loss = np.mean(train_losses)

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()
        
        # Logging (example for total_loss, adjust as needed)
        print(f'Epoch: {epoch}/{final_epoch}, Batch: {i+1}/{len(train_loader)}, Loss: {total_loss.item()}')

    # Validation loop
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_data in val_loader:
            pCT_val, rCT_val = batch_data["plan"].to(device), batch_data["repeat"].to(device)
            reg_val = batch_data["pos"].to(device)  # Ground truth coordinates
            
            # Model prediction
            outx_val, outy_val, outz_val = model(pCT_val, rCT_val)
       
            # Split reg_val into its components
            regx_val, regy_val, regz_val = reg_val[:, 0], reg_val[:, 1], reg_val[:, 2]
          
            # Calculate validation loss for each component
            loss_val_x = mae_loss(outx_val, regx_val.unsqueeze(1))
            loss_val_y = mae_loss(outy_val, regy_val.unsqueeze(1))
            loss_val_z = mae_loss(outz_val, regz_val.unsqueeze(1))
            
            # Combine the losses
            total_val_loss = (loss_val_x + loss_val_y + loss_val_z)/batch_size
            
            val_losses.append(total_val_loss.item())
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch [{epoch+1}/{final_epoch}], Validation Loss: {avg_val_loss:.4f}')


        # Adjust learning rate
        scheduler.step(avg_val_loss)
        
        


        # Saving model and logging
        save_dir = '/home/shahpouriz/Data/DBP_Project/LOG'
        filename = f'loss_Model_3f_newopt_morebatches'
        loss_file = os.path.join(save_dir, f'{filename}.txt')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        if avg_val_loss <= best_mae and epoch > 0:
            best_mae = avg_val_loss
            model_filename = f'{filename}_{epoch+1}.pt'  # Store the model filename
            torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
            print(f'Model saved to {os.path.join(save_dir, model_filename)}')
        
        # Append current epoch's average loss and validation loss to the log file
        with open(loss_file, 'a') as f:
            f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n')
            # f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {avg_train_loss}, Val: {mean_val_loss}\n')
