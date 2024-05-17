
import os
import numpy as np
import torch
from tqdm import tqdm
import json
from dual_network import Dual3DCNN3, Dual3DCNN4, Dual3DCNN5, Dual3DCNN6
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
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, RandGaussianNoised, SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged, RandGibbsNoised, RandKSpaceSpikeNoised, RandRicianNoised, ScaleIntensityRanged
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
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import NormalizeIntensityd
from utilities import get_date

scaler = GradScaler()  # For mixed precision training

# data_path_NEW = '/data/shahpouriz/DBP_newDATA/nrrd/oneCTperPatients/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneCTperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneOPTZperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneoneperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_DATA_total/nrrd/oneCTperPatinet/proton'
# data_path = '/data/shahpouriz/DBP_CTs/nrrd/proton'
data_path = '/data/shahpouriz/Processed_CT/nrrd/proton'

patient_list_NEW = list_patient_folders(data_path)
# Shuffle patient list if you want randomness
random.seed(42)  # You can choose any number as the seed
random.shuffle(patient_list_NEW)

# Define split sizes
total_patients = len(patient_list_NEW)
train_size = int(total_patients * 0.80)
val_size = int(total_patients * 0.10)
# The rest will be for the test set

# Split the patient list
train_patients = patient_list_NEW[:train_size]
val_patients = patient_list_NEW[train_size:train_size + val_size]
test_patients = patient_list_NEW[train_size + val_size:]

train_pct, train_rct, train_pos = prepare_data_nrrd_for_CT(data_path, train_patients)
val_pct, val_rct, val_pos = prepare_data_nrrd_for_CT(data_path, val_patients)
test_pct, test_rct, test_pos = prepare_data_nrrd_for_CT(data_path, test_patients)

# Create dictionaries for each dataset
train_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(train_pct, train_rct, train_pos)]
val_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(val_pct, val_rct, val_pos)]
test_data = [{"plan": img, "repeat": tar, "pos": pos} for img, tar, pos in zip(test_pct, test_rct, test_pos)]

# Check the lengths of the sets
print("Number of training samples:", len(train_data))
print("Number of validation samples:", len(val_data))
print("Number of test samples:", len(test_data))
print(len(test_data)+len(val_data)+len(train_data))


dim = 96
pix = 2.0
size = (dim, dim, dim)
pixdim = (pix, pix, pix)
spacing_mode = 'trilinear'
spacial_pad = 'constant'
reader = ITKReader()
transforms = Compose([
        LoadImaged(keys=["plan", "repeat"], reader=reader),
        EnsureChannelFirstd(keys=["plan", "repeat"]),
        NormalizeIntensityd(keys=["plan", "repeat"]),

        Spacingd(keys=["plan", "repeat"], pixdim=pixdim, mode=spacing_mode),
        SpatialPadd(keys=["plan", "repeat"], spatial_size=size, mode=spacial_pad),  # Ensure minimum size
        CenterSpatialCropd(keys=["plan", "repeat"], roi_size=size),  # Ensure uniform size
        # ScaleIntensityRanged(keys=["plan", "repeat"], a_min=-800, a_max=1600, b_min=0, b_max=1, clip=True),

        # RandStdShiftIntensity(factors = (5 , 10), prob=0.1, nonzero=False, channel_wise=False
        # RandBiasField(degree=3, coeff_range=(0.0, 0.1), prob=0.1),
        # SavitzkyGolaySmooth(window_length, order, axis=1, mode='zeros'),
        # MedianSmooth(radius=1)
        # GaussianSmooth(sigma=1.0, approx='erf')
        # RandGaussianSmooth(sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), prob=0.1, approx='erf'),
        # GaussianSharpen(sigma1=3.0, sigma2=1.0, alpha=30.0, approx='erf'),
        # RandGaussianSharpen(sigma1_x=(0.5, 1.0), sigma1_y=(0.5, 1.0), sigma1_z=(0.5, 1.0), sigma2_x=0.5, sigma2_y=0.5, sigma2_z=0.5, alpha=(10.0, 30.0), approx='erf', prob=0.1)
        # RandCoarseShuffle(holes, spatial_size, max_holes=None, max_spatial_size=None, prob=0.1),
        # RandCoarseDropout(holes, spatial_size, dropout_holes=True, fill_value=None, max_holes=None, max_spatial_size=None, prob=0.1)
        
        # RandGibbsNoised(keys=["plan", "repeat"], prob=0.2, alpha=(0.0, 1.0)),
        # RandGaussianNoised(keys=["plan", "repeat"], prob=0.2, mean=0.0, std=0.1),
        # RandKSpaceSpikeNoised(keys=["plan", "repeat"], prob=0.2, intensity_range=None, channel_wise=True),
        # RandRicianNoised(keys=["plan", "repeat"], prob=0.2, mean=1.0, std=0.5),
    ])

batch_size=50
cache_rate = 0.8 #0.5 # 0.01 #
num_workers = 5
train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)



lambda_reg = None, # 1e-4  # for L1 regularization
weight_decay = None, # 1e-4  # for L2 regularization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_epoch = 50
learning_rate = 0.01
best_mae = np.inf


save_dir = '/home/shahpouriz/Data/DBP_Project/LOG_CT'
filename = get_date()
loss_file = f'{save_dir}/loss_{filename}.txt'
path_experiments = f'{save_dir}/experiments_{filename}.json'  # Path to save the JSON file


if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


model =Dual3DCNN5(width=dim, height=dim, depth=dim).to(device)
num_filters = model.num_filters
kernel_size = model.kernel_size
stride = model.stride
dropout = model.dropout
initializer = model.initializer

# loss_func = torch.nn.MSELoss()
loss_func = torch.nn.L1Loss()
# loss_func = np.square(torch.nn.MSELoss().item())

  # Start with a learning rate
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)


################################################################

if os.path.exists(path_experiments):
    with open(path_experiments, 'r') as file:
        data = json.load(file)
else:
    data = dict()

from utilities import save_experiment_details
data = save_experiment_details(data, path_experiments, 
                               train_data, val_data, test_data,
                               final_epoch, optimizer, scheduler, dim, pixdim,
                               batch_size, cache_rate, num_workers, 
                               reader, spacing_mode, spacial_pad,
                               initializer, num_filters, kernel_size, stride, dropout,
                               learning_rate, lambda_reg, weight_decay, weight_correction=None)
                               # df, df_loss_min, df_val_min, )

with open(path_experiments, 'w') as file:
        json.dump(data, file, indent=4)


################################################################
model.train()
for epoch in range(final_epoch):

    model.train()
    loss_list = []
    for i, batch_data in enumerate(train_loader):
        pCT, rCT = batch_data["plan"].to(device), batch_data["repeat"].to(device)
        reg = batch_data["pos"].clone().to(device)  # If gradients are required for 'reg'
        print("------------------------")
        print(f"reg: ",{reg})
        
        optimizer.zero_grad()

        output = model(pCT, rCT)
        print(output)

        loss = loss_func(output, reg)


        # L1 Regularization
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            if 'weight' in name:  # Apply L1 only to weights, not biases
                l1_reg = l1_reg + torch.norm(param, 1)
        
        # Combine loss with L1 regularization
        # loss = loss  + lambda_reg * l1_reg


        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        mean_tot_loss = np.mean(loss_list)
        print(f'Epoch: {epoch}/{final_epoch}, Batch: {i+1}/{len(train_loader)}, Loss_avg: {mean_tot_loss}')


    # Validation loop
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_data in val_loader:
            pCT_val, rCT_val = batch_data["plan"].to(device), batch_data["repeat"].to(device)
            reg_val = batch_data["pos"].clone().to(device)  # If gradients are required for 'reg'
            print(reg_val)

            output_val = model(pCT_val, rCT_val)
            print(output_val)
            loss_output_val = loss_func(output_val, reg_val)

            val_loss.append(loss_output_val.item())

        mean_val_loss = np.mean(val_loss)
        print(f'Epoch [{epoch+1}/{final_epoch}], Validation Loss: {mean_val_loss:.4f}')

        # Adjust learning rate
        scheduler.step(mean_val_loss)
   
    current_valid_mae = mean_val_loss



    with open(loss_file, 'a') as f: #a-append
        f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {mean_tot_loss}, Val: {mean_val_loss}\n')
        if current_valid_mae <= best_mae and epoch > 0:
            best_mae = current_valid_mae
            model_filename = f'{filename}.pt'  # Store the model filename
            torch.save(model.state_dict(),f'{save_dir}/{model_filename}')
            f.write(f'{model_filename} is saved! for epoch {epoch+1}\n')  
            print(f'{model_filename} is saved! for epoch {epoch+1}\n')  


epochs = []
training_losses = []
validation_losses = []
start_epoch = 0  # Define the start epoch from which you want to plot

with open(loss_file, 'r') as file:
    for line in file:
        match_epoch = re.search(r'Epoch: (\d+)/\d+, Loss: (\d+\.\d+), Val: (\d+\.\d+)', line)
        if match_epoch:
            epoch_num = int(match_epoch.group(1))
            if epoch_num >= start_epoch:  # Only add data from the start_epoch onwards
                epochs.append(epoch_num)
                training_losses.append(float(match_epoch.group(2)))
                validation_losses.append(float(match_epoch.group(3)))

import pandas as pd
df = pd.read_csv(loss_file, sep=',')
df = pd.read_csv(loss_file, header=None, sep=",", names=["Epoch_info", "Loss", "Validation"])

df = df.dropna()
df['loss'] = df['Loss'].str.split(':').str[-1].astype(float)
df['val'] = df['Validation'].str.split(':').str[-1].astype(float)

df = df.reset_index().drop(columns=['index'])
df = df[(df.Val < 20) & (df.Loss < 20)]

df_loss_min = df[df.Loss == df.Loss.min()]
df_val_min = df[df.Val == df.val.min()]


#######################################################################


    
    