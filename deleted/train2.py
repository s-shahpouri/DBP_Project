
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
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


data_path_NEW = '/home/shahpouriz/Data/DBP_newDATA/DBP/nrrd/test'
data_path_NEW = '/home/shahpouriz/Data/DBP_newDATA/new_patients_sama/nrrd/oneCTperPatients'

patient_list_NEW = list_patient_folders(data_path_NEW)
# # Shuffle patient list if you want randomness
# 
random.shuffle(patient_list_NEW)

# Define split sizes
total_patients = len(patient_list_NEW)
train_size = int(total_patients * 0.70)
val_size = int(total_patients * 0.20)
# The rest will be for the test set

print(total_patients, train_size, val_size)
# Split the patient list
train_patients = patient_list_NEW[:train_size]
val_patients = patient_list_NEW[train_size:train_size + val_size]
test_patients = patient_list_NEW[train_size + val_size:]

train_pct, train_rct, train_pos = prepare_data_nrrd(data_path_NEW, train_patients)
val_pct, val_rct, val_pos = prepare_data_nrrd(data_path_NEW, val_patients)
test_pct, test_rct, test_pos = prepare_data_nrrd(data_path_NEW, test_patients)

# Create dictionaries for each dataset
train_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(train_pct, train_rct, train_pos)]
val_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(val_pct, val_rct, val_pos)]
test_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(test_pct, test_rct, test_pos)]


# Check the lengths of the sets
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of test samples:", len(test_data))
print(len(test_data)+len(val_data)+len(train_data))

# Set parameters
starting_epoch = 0
final_epoch = 50

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


train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=0.9, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=0.9, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

# Build model
print('Initializing model...')
model = Dual3DCNN5(width=dim, height=dim, depth=dim)
device = torch.device("cuda:0")
model.to(device)

# mae_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()


learning_rate =  0.0001 #0.0002332 #0.000538 #1e-4  # Start with a learning rate 0.0002332795188029022
lambda_reg = 0.000001

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

        # L1 Regularization
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        
        loss_output += lambda_reg * l1_reg

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
    filename = f'model_1rCT'
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
