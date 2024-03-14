
import os
import numpy as np
import torch
from tqdm import tqdm
import json
from dual_network import Dual3DCNN3, Dual3DCNN4, Dual3DCNN5
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



data_path_NEW = '/home/shahpouriz/Data/DBP_newDATA/DBP/nrrd/test'
# data_path_OLD = '/home/shahpouriz/Data/DBP_oldDATA/nrrd/proton'
# # Get the list of patient folders

# patient_list_NEW = list_patient_folders(data_path_NEW)
# pct, rct, pos = prepare_data_nrrd(data_path_NEW, patient_list_NEW)
# data_NEW = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct, rct, pos)]

# patient_list_OLD = list_patient_folders(data_path_OLD)
# pct, rct, pos = prepare_data_nrrd(data_path_OLD, patient_list_OLD)
# data_OLD = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct, rct, pos)]


# # Assuming data_NEW and data_OLD are your lists of dictionaries
# data = data_NEW + data_OLD
# # data = data_NEW #[:20] 

# # Split the data
# train_data, val_data, test_data = split_data(data)

# # Check the lengths of the sets
# print("Number of training samples:", len(train_data))
# print("Number of validation samples:", len(val_data))
# print("Number of test samples:", len(test_data))

import random
patient_list_NEW = list_patient_folders(data_path_NEW)
# Shuffle patient list if you want randomness
random.shuffle(patient_list_NEW)

# Define split sizes
total_patients = len(patient_list_NEW)
train_size = int(total_patients * 0.7)
val_size = int(total_patients * 0.20)
# The rest will be for the test set

# Split the patient list
train_patients = patient_list_NEW[:train_size]
val_patients = patient_list_NEW[train_size:train_size + val_size]
test_patients = patient_list_NEW[train_size + val_size:]

# Now you can prepare your data
# This step will depend on how your 'prepare_data_nrrd' function works
# You need to pass the right patient list to this function for each set
train_pct, train_rct, train_pos = prepare_data_nrrd(data_path_NEW, train_patients)
val_pct, val_rct, val_pos = prepare_data_nrrd(data_path_NEW, val_patients)
test_pct, test_rct, test_pos = prepare_data_nrrd(data_path_NEW, test_patients)

# Create dictionaries for each dataset
train_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(train_pct, train_rct, train_pos)]
val_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(val_pct, val_rct, val_pos)]
test_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(test_pct, test_rct, test_pos)]



# Set parameters
starting_epoch = 0
final_epoch = 200

# Condition for saving list
best_mae = np.inf
exception_list = ['']


dim = 128
size = (dim, dim, dim)
pixdim = (3.0, 3.0, 3.0)
transforms = Compose([
        LoadImaged(keys=["plan", "repeat"], reader=ITKReader()),
        EnsureChannelFirstd(keys=["plan", "repeat"]),
        ScaleIntensityd(keys=["plan", "repeat"]),
        Spacingd(keys=["plan", "repeat"], pixdim=pixdim, mode='trilinear'),
        SpatialPadd(keys=["plan", "repeat"], spatial_size=size, mode='constant'),  # Ensure minimum size
        CenterSpatialCropd(keys=["plan", "repeat"], roi_size=size),  # Ensure uniform size
    ])


train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=1.0, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=1.0, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

# train_ds = SmartCacheDataset(data=train_data, transform=transforms, cache_rate=0.1, replace_rate=0.2, num_init_workers=1, num_replace_workers=1)
# train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

# val_ds = SmartCacheDataset(data=val_data, transform=transforms, cache_rate=0.1, replace_rate=0.2, num_init_workers=1, num_replace_workers=1)
# val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)


# Build model
print('Initializing model...')
model = Dual3DCNN5(width=dim, height=dim, depth=dim)
device = torch.device("cuda:0")
model.to(device)

# mae_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()

from torch.optim.lr_scheduler import ReduceLROnPlateau
learning_rate = 1e-4  # Start with a learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)



# Training loop
for epoch in range(starting_epoch, final_epoch):
    model.train()  # Set model to training mode
    mae_list = []
    train_loss = []
    for i, batch_data in enumerate(train_loader):  # Use enumerate to get the batch index
        pCT, rCT = batch_data["plan"].to(device), batch_data["repeat"].to(device)
        reg = batch_data["pos"].clone().detach().requires_grad_(True).to(device)  # If gradients are required for 'reg'
        optimizer.zero_grad()

        output = model(pCT, rCT)
        loss_output = mae_loss(output, reg)

        loss_output.backward()
        optimizer.step()
        
        # Logging
        mae_list.append(loss_output.item())
        mean_mae = np.mean(mae_list)
        # Corrected to print the current batch number
        print(f'Epoch: {epoch}/{final_epoch}, Batch: {i+1}/{len(train_loader)}, Loss_avg: {mean_mae}')

    # Validation loop
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_data in val_loader:
            pCT_val, rCT_val = batch_data["plan"].to(device), batch_data["repeat"].to(device)
            reg_val = batch_data["pos"].clone().detach().requires_grad_(True).to(device)  # If gradients are required for 'reg'

            output_val = model(pCT_val, rCT_val)
            loss_output_val = mae_loss(output_val, reg_val)

            val_loss.append(loss_output_val.item())

        mean_val_loss = np.mean(val_loss)
        print(f'Epoch [{epoch+1}/{final_epoch}], Validation Loss: {mean_val_loss:.4f}')

        # Adjust learning rate
        scheduler.step(mean_val_loss)
        
    save_dir = '/home/shahpouriz/Data/DBP_Project/LOG'
    filename = f'model_{dim}_all_opt_deepermodel'
    loss_file = fr'/home/shahpouriz/Data/DBP_Project/LOG/{filename}.txt'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    current_valid_mae = mean_val_loss

    with open(loss_file, 'a') as f: #a-append
        f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {mean_mae}, Val: {mean_val_loss}\n')
        if current_valid_mae <= best_mae and epoch > 0:
            best_mae = current_valid_mae
            model_filename = f'{filename}_{epoch+1}.pt'  # Store the model filename
            torch.save(model.state_dict(),f'{save_dir}/{model_filename}')
            f.write(f'{model_filename} is saved!\n')  # This line should be inside the 'with' block
