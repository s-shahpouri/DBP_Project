import os
import numpy as np
import torch
from tqdm import tqdm
import json
from dual_network_3f import Dual3DCNN6 as Dual_3f
from dual_network_3f import FlexibleDual3DCNN_3f as FlexibleDual_3f

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
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import LoadImaged
from monai.data.image_reader import ITKReader
from utilities import list_patient_folders, prepare_data_nrrd, split_data
from torch.optim.lr_scheduler import ReduceLROnPlateau


data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneCTperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneOPTZperPatinet/proton'

patient_list_NEW = list_patient_folders(data_path)
# Shuffle patient list if you want randomness
random.seed(42)  # You can choose any number as the seed
random.shuffle(patient_list_NEW)

# Define split sizes
total_patients = len(patient_list_NEW)
train_size = int(total_patients * 0.70)
val_size = int(total_patients * 0.20)
# The rest will be for the test set

# Split the patient list
train_patients = patient_list_NEW[:train_size]
val_patients = patient_list_NEW[train_size:train_size + val_size]
test_patients = patient_list_NEW[train_size + val_size:]

train_pct, train_rct, train_pos = prepare_data_nrrd(data_path, train_patients)
val_pct, val_rct, val_pos = prepare_data_nrrd(data_path, val_patients)
test_pct, test_rct, test_pos = prepare_data_nrrd(data_path, test_patients)

# Create dictionaries for each dataset
train_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(train_pct, train_rct, train_pos)]
val_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(val_pct, val_rct, val_pos)]
test_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(test_pct, test_rct, test_pos)]


# Check the lengths of the sets
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of test samples:", len(test_data))
print(len(test_data)+len(val_data)+len(train_data))


batch_size = 1

dim = 128
size = (dim, dim, dim)
pixdim = (3.0, 3.0, 3.0)
transforms = Compose([
        LoadImaged(keys=["plan", "repeat"], reader=ITKReader()),
        EnsureChannelFirstd(keys=["plan", "repeat"]),
        # ScaleIntensityd(keys=["plan", "repeat"]),
        Spacingd(keys=["plan", "repeat"], pixdim=pixdim, mode='trilinear'),
        SpatialPadd(keys=["plan", "repeat"], spatial_size=size, mode='constant'),  # Ensure minimum size
        CenterSpatialCropd(keys=["plan", "repeat"], roi_size=size),  # Ensure uniform size
    ])


train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=0.8, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)

val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=0.8, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)


lambda_reg = 1e-04 # for L1 regularization
weight_decay = 1e-04  # for L2 regularization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_epoch = 30
learning_rate = 0.0001
best_mae = np.inf 
architectures = [32, 64, 128, 256]

save_dir = '/home/shahpouriz/Data/DBP_Project/LOG_3f_1ct'
filename = f'{dim}_1ctP_{learning_rate}_{architectures}_L1L2(4)'
loss_file = fr'/home/shahpouriz/Data/DBP_Project/LOG_3f_1ct/{filename}.txt'

if not os.path.isdir(save_dir):
        os.makedirs(save_dir)



model = FlexibleDual_3f(architectures)
device = torch.device("cuda:0")
model.to(device)


loss_func = torch.nn.L1Loss()


# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)




# Initialize best_mae before the training loop
best_mae = float('inf')
train_losses = []

# Training loop
for epoch in range(final_epoch):
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
        loss_x = loss_func(outx, regx.unsqueeze(1))  # Add dimension to match output shape
        loss_y = loss_func(outy, regy.unsqueeze(1))
        loss_z = loss_func(outz, regz.unsqueeze(1))

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
        # print(f'Epoch: {epoch}/{final_epoch}, Batch: {i+1}/{len(train_loader)}, Loss: {total_loss.item()}')

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
            loss_val_x = loss_func(outx_val, regx_val.unsqueeze(1))
            loss_val_y = loss_func(outy_val, regy_val.unsqueeze(1))
            loss_val_z = loss_func(outz_val, regz_val.unsqueeze(1))
            
            # Combine the losses
            total_val_loss = (loss_val_x + loss_val_y + loss_val_z)/batch_size
            
            val_losses.append(total_val_loss.item())
        
        # Calculate average validation loss
        avg_val_loss = np.mean(val_losses)
        print(f'Epoch [{epoch+1}/{final_epoch}], Validation Loss: {avg_val_loss:.4f}')


        # Adjust learning rate
        scheduler.step(avg_val_loss)
        
        

    current_valid_mae = avg_val_loss

    with open(loss_file, 'a') as f: #a-append
        f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {avg_train_loss}, Val: {avg_val_loss}\n')
        if current_valid_mae <= best_mae and epoch > 0:
            best_mae = current_valid_mae
            model_filename = f'{filename}.pt'  # Store the model filename
            torch.save(model.state_dict(),f'{save_dir}/{model_filename}')
            f.write(f'{model_filename} is saved! for epoch {epoch+1}\n')  
            print(f'{model_filename} is saved! for epoch {epoch+1}\n')  
    
        # if avg_val_loss <= best_mae and epoch > 0:
        #     best_mae = avg_val_loss
        #     model_filename = f'{filename}_{epoch+1}.pt'  # Store the model filename
        #     torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
        #     print(f'Model saved to {os.path.join(save_dir, model_filename)}')
        
        # # Append current epoch's average loss and validation loss to the log file
        # with open(loss_file, 'a') as f:
        #     f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n')
        #     # f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {avg_train_loss}, Val: {mean_val_loss}\n')
