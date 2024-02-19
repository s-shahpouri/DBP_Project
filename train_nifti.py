
import os
import fnmatch
import numpy as np
import torch
from tqdm import tqdm
import random
import json
from torch.utils.data import DataLoader
import glob
import os
import json

from dual_network import Dual3DCNN6 as Dual
# from Dataset_json import PXAI_Dataset
from decayLR import DecayLR
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import glob
from utilities import create_list_from_master_json, read_json_file, split_data
import os
import glob
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader



from utilities import list_patient_folders, prepare_data

# Specify the directory where the patient folders are located
data_path = '/home/shahpouriz/Data/DBP_newDATA/DBP/nifti/proton'

# Get the list of patient folders
patient_list = list_patient_folders(data_path)

# Prepare training, validation, and testing datasets
pct_train, rct_train, pos_train = prepare_data(data_path, patient_list)
pct_val, rct_val, pos_val = prepare_data(data_path, patient_list)
pct_test, rct_test, pos_test = prepare_data(data_path, patient_list)


# Set parameters
starting_epoch = 0
decay_epoch = 20
final_epoch = 30
learning_rate = 0.0001
batchsize = 5
device_num = 1
lambda_reg = 0.000001

# Condition for saving list
save_list = False
best_mae = np.inf

exception_list = ['']



train_dict = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct_train, rct_train, pos_train)]

val_dict = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct_val, rct_val, pos_val)]

test_dict = [{"plan": img[0], "repeat": tar, "pos": pos} for img, tar, pos in zip(pct_test, rct_test, pos_test)]

train_files = train_dict[-100:]
val_files = val_dict[-20:]

#### My method

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import LoadImaged
from monai.data.image_reader import ITKReader


# Assuming the desired input size for the model is [96, 96, 96]
desired_size = [256, 256, 256]

transforms = Compose([
        LoadImaged(keys=["plan", "repeat"], reader=ITKReader()),
        
        EnsureChannelFirstd(keys=["plan", "repeat"]),
        ScaleIntensityRanged(
            keys=["plan", "repeat"],
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Spacingd(keys=["plan", "repeat"], pixdim=(3.0, 3.0, 3.0), mode='trilinear'),
        SpatialPadd(keys=["plan", "repeat"], spatial_size=(128, 128, 128), mode='constant'),  # Ensure minimum size
        CenterSpatialCropd(keys=["plan", "repeat"], roi_size=(128, 128, 128)),  # Ensure uniform size
    ])


train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, SpatialPadd, CenterSpatialCropd, ScaleIntensityRanged
# from monai.data import CacheDataset, DataLoader, Dataset
# import matplotlib.pyplot as plt

# # Assuming `transforms` is defined and `val_files` contains your validation files
# check_ds = Dataset(data=val_files, transform=transforms)
# check_loader = DataLoader(check_ds, batch_size=1)

# # Manually retrieve the first batch of data
# for check_data in check_loader:
#     break

# plan, repeat = (check_data["plan"][0][0], check_data["repeat"][0][0])
# print(f"image shape: {plan.shape}, target shape: {repeat.shape}")

# # plot the slice [:, :, n]
# n = 80

# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(plan[:, :, n])
# plt.subplot(1, 2, 2)
# plt.title("repeat")
# plt.imshow(repeat[:, :, n])
# plt.show()



# Build model
print('Initializing model...')
model = Dual(width=512, height=512, depth=512)
device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss
print('Defining loss...')
mae_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()

# Define optimizer
print('Defining optimizer...')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Define scheduler
print('Defining scheduler...')
lr_lambda = DecayLR(epochs=final_epoch, offset=0, decay_epochs=decay_epoch).step
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# Training loop
for epoch in range(starting_epoch, final_epoch):
    model.train()  # Set model to training mode
    mae_list = []
    train_loss = []
    for i, batch_data in enumerate(train_loader):  # Use enumerate to get the batch index
        pCT, rCT = batch_data["plan"].to(device), batch_data["repeat"].to(device)
        reg = batch_data["pos"].clone().detach().requires_grad_(True).to(device)  # If gradients are required for 'reg'


        # Forward pass
        output = model(pCT, rCT)
        loss_output = mse_loss(output, reg)
            
        # L1 Regularization
        l1_reg = torch.tensor(0., requires_grad=True).to(device)
        for name, param in model.named_parameters():
            l1_reg = l1_reg + torch.norm(param, 1)
        loss_output += lambda_reg * l1_reg
        
        # Backpropagation
        optimizer.zero_grad()  
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
            loss_val = mae_loss(output_val, reg_val)

            val_loss.append(loss_val.item())

            mean_val_loss = np.mean(val_loss)
            print(f'Epoch [{epoch+1}/{final_epoch}], Validation Loss: {mean_val_loss:.4f}')

    # Adjust learning rate
    lr_scheduler.step(mean_val_loss)
    
    save_dir = '/home/shahpouriz/Data/DBP_Project/LOG'
    loss_file = fr'/home/shahpouriz/Data/DBP_Project/LOG/loss_dose_json_simpleModel.txt'

    # Save model
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    current_valid_mae = val_loss[-1]    
    if current_valid_mae <= best_mae and epoch > 0:
        best_mae = current_valid_mae
        torch.save(model.state_dict(),f'{save_dir}/model_weights_dose_{epoch+1}.pt')
    with open(loss_file, 'a') as f: #a-append
        f.write(f'Epoch: {epoch+1}/{final_epoch}, Loss: {mean_mae}, Val: {mean_val_loss}\n')
