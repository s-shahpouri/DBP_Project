
import os
import numpy as np
import torch
from tqdm import tqdm
import json
from dual_network_3f import Dual3DCNN6_3f
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
from utilities import list_patient_folders, prepare_data_nrrd, split_data, prepare_data_nrrd_for_CT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
import functools
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()  # For mixed precision training

# data_path_NEW = '/data/shahpouriz/DBP_newDATA/nrrd/oneCTperPatients/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneCTperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneOPTZperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneoneperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneCTperPatinet/proton'
data_path = '/data/shahpouriz/DBP_CTs/nrrd/proton'


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

train_pct, train_rct, train_pos = prepare_data_nrrd_for_CT(data_path, train_patients)
val_pct, val_rct, val_pos = prepare_data_nrrd_for_CT(data_path, val_patients)
test_pct, test_rct, test_pos = prepare_data_nrrd_for_CT(data_path, test_patients)

# Create dictionaries for each dataset
import numpy as np

# Create dictionaries for each dataset with pos values as NumPy arrays divided by 3
train_data = [{"plan": img, "repeat": tar, "pos": np.array(pos) / 3} for img, tar, pos in zip(train_pct, train_rct, train_pos)]
val_data = [{"plan": img, "repeat": tar, "pos": np.array(pos) / 3} for img, tar, pos in zip(val_pct, val_rct, val_pos)]
test_data = [{"plan": img, "repeat": tar, "pos": np.array(pos) / 3} for img, tar, pos in zip(test_pct, test_rct, test_pos)]


# Check the lengths of the sets
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of test samples:", len(test_data))
print(len(test_data)+len(val_data)+len(train_data))


from monai.transforms import NormalizeIntensityd

dim = 512
size = (dim, dim, dim)
pixdim = (1.0, 1.0, 1.0)
transforms = Compose([
        LoadImaged(keys=["plan", "repeat"], reader=ITKReader()),
        EnsureChannelFirstd(keys=["plan", "repeat"]),
        NormalizeIntensityd(keys=["plan", "repeat"]),
        Spacingd(keys=["plan", "repeat"], pixdim=pixdim, mode='trilinear'),
        SpatialPadd(keys=["plan", "repeat"], spatial_size=size, mode='constant'),  # Ensure minimum size
        CenterSpatialCropd(keys=["plan", "repeat"], roi_size=size),  # Ensure uniform size
    ])


train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=0.01, num_workers=1)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)

val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=0.1, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)



lambda_reg = 1e-5  # for L1 regularization
weight_decay = 1e-5  # for L2 regularization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


final_epoch = 100
learning_rate = 0.0001
best_mae = np.inf


# Build model



save_dir = '/home/shahpouriz/Data/DBP_Project/LOG_CT'
filename = f'cnn6_3f_{dim}_{learning_rate}_L1L2(5)_sn'
loss_file = f'{save_dir}/{filename}.txt'

if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


model =Dual3DCNN6_3f(width=dim, height=dim, depth=dim).to(device)
print("Model device:", next(model.parameters()).device)

# mae_loss = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()


  # Start with a learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)


batch_size = 1

train_losses = []
val_losses = []

for epoch in range(final_epoch):
    model.train()
    mae_list = []
    batch_losses = []
    for i, batch_data in enumerate(train_loader):
        pCT, rCT = batch_data["plan"].to(device), batch_data["repeat"].to(device)
        reg = batch_data["pos"].clone().to(device)  # If gradients are required for 'reg'
        print("------------------------")
        print(f"reg: ",{reg})
        optimizer.zero_grad()

        with autocast():  # Enable automatic mixed precision
            outx, outy, outz = model(pCT, rCT)  # Model predictions
            # Compute losses for each output dimension
            loss_x = loss_func(outx, reg[:, 0].unsqueeze(1))
            loss_y = loss_func(outy, reg[:, 1].unsqueeze(1))
            loss_z = loss_func(outz, reg[:, 2].unsqueeze(1))
            loss = (loss_x + loss_y + loss_z) / 3  # Average the losses

        scaler.scale(loss).backward()  # Scale the loss and perform backward propagation through the scaled loss
        scaler.step(optimizer)  # Update model parameters
        scaler.update()  # Update the scaler

        batch_losses.append(loss.item())  # Append the loss of the current batch to the list
        if i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')

    epoch_loss = np.mean(batch_losses)  # Average loss for this epoch
    train_losses.append(epoch_loss)  # Append the average loss of this epoch to the global list


 
    # Validation loop
    model.eval()
    batch_val_losses = []  # Store validation losses for each batch

    with torch.no_grad():
        for batch_data in val_loader:
            pCT_val, rCT_val = batch_data["plan"].to(device), batch_data["repeat"].to(device)
            reg_val = batch_data["pos"].to(device)  # Ground truth coordinates
            
            with autocast():
                outx_val, outy_val, outz_val = model(pCT_val, rCT_val)
                loss_val_x = loss_func(outx_val, reg_val[:, 0].unsqueeze(1))
                loss_val_y = loss_func(outy_val, reg_val[:, 1].unsqueeze(1))
                loss_val_z = loss_func(outz_val, reg_val[:, 2].unsqueeze(1))
                val_loss = (loss_val_x + loss_val_y + loss_val_z) / 3
            val_losses.append(val_loss.item())
 
            batch_val_losses.append(val_loss.item())  # Append each validation loss to the list

    epoch_val_loss = np.mean(batch_val_losses)  # Average validation loss for this epoch
    val_losses.append(epoch_val_loss)  # Append the average validation loss of this epoch to the global list

    print(f'Epoch {epoch+1}/{final_epoch}, Training Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

    # Adjust learning rate based on the average validation loss
    scheduler.step(epoch_val_loss)

    # Save the model if this epoch's validation loss is the best so far
    if epoch_val_loss < best_mae:
        best_mae = epoch_val_loss
        model_filename = f'{filename}_{epoch+1}.pt'
        torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
        print(f'Model saved as {model_filename} at epoch {epoch+1}')


    # with open(loss_file, 'a') as f: #a-append
    #     f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {avg_train_loss}, Val: {mean_val_loss}\n')
    #     if current_valid_mae <= best_mae and epoch > 0:
    #         best_mae = current_valid_mae
    #         model_filename = f'{filename}.pt'  # Store the model filename
    #         torch.save(model.state_dict(),f'{save_dir}/{model_filename}')
    #         f.write(f'{model_filename} is saved! for epoch {epoch+1}\n')  
    #         print(f'{model_filename} is saved! for epoch {epoch+1}\n')  
