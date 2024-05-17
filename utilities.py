
import json
import os
import random
import re
import glob
import numpy as np
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, Spacingd, SpatialPadd, CenterSpatialCropd


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_list_from_master_json(test_json_path, data_path, exception_list=[]):
    json_data = read_json_file(test_json_path)

    px_ID = json_data['ID']
    
    if px_ID in exception_list:
        pCT_dose_list = []
        rCT_dose_list = []
        final_translation_list = []
        return pCT_dose_list, rCT_dose_list, final_translation_list    
    
    else:
        num_of_plans = len(json_data['plans'])

        pCT_dose_list = []
        rCT_dose_list = []
        final_translation_list = []

        for i in range(num_of_plans):
            plan_name = json_data['plans'][i]['name']
            pCT_dose_path = json_data['plans'][i]['planning_examination']['dose_DICOM_filename']
            
            directory,pCT_dose_fn = os.path.split(pCT_dose_path)
            name, ext = os.path.splitext(pCT_dose_fn)
            pCT_dose_nrrd_resampled = os.path.join(data_path,directory,name+'_resampled.nrrd')
            
            num_of_evals = len(json_data['plans'][i]['evaluation_examinations'])
            for k in range(num_of_evals):
                rct_name = json_data['plans'][i]['evaluation_examinations'][k]['name']
                
                if rct_name != 'pCTp0': # Exclude pCTp0 - WIP
                    num_of_opt_case = len(json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'])
                    for l in range(num_of_opt_case):
                        if json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'][l]['completed'] == True:
                            rct_dose_path = json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'][l]['gradient_descent_result']['dose_DICOM_filename']
                            
                            directory2, rct_dose_fn = os.path.split(rct_dose_path)
                            name2, ext2 = os.path.splitext(rct_dose_fn)
                            rct_dose_nrrd_resampled = os.path.join(data_path,directory2,name2+'_resampled.nrrd')
                            
                            final_translation_coordinate = json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'][l]['gradient_descent_result']['final_translation_coordinate']
                            final_coordinates = [final_translation_coordinate['x'],final_translation_coordinate['y'],final_translation_coordinate['z']]
                            pCT_dose_list.append(pCT_dose_nrrd_resampled)
                            rCT_dose_list.append(rct_dose_nrrd_resampled)
                            final_translation_list.append(final_coordinates)
        
        return pCT_dose_list, rCT_dose_list, final_translation_list


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_list_from_master_json(test_json_path, data_path, exception_list=[]):
    json_data = read_json_file(test_json_path)

    px_ID = json_data['ID']
    
    if px_ID in exception_list:
        pCT_dose_list = []
        rCT_dose_list = []
        final_translation_list = []
        return pCT_dose_list, rCT_dose_list, final_translation_list    
    
    else:
        num_of_plans = len(json_data['plans'])

        pCT_dose_list = []
        rCT_dose_list = []
        final_translation_list = []

        for i in range(num_of_plans):
            plan_name = json_data['plans'][i]['name']
            pCT_dose_path = json_data['plans'][i]['planning_examination']['dose_DICOM_filename']
            
            directory,pCT_dose_fn = os.path.split(pCT_dose_path)
            name, ext = os.path.splitext(pCT_dose_fn)
            pCT_dose_nrrd_resampled = os.path.join(data_path,directory,name+'_resampled.nrrd')
            
            num_of_evals = len(json_data['plans'][i]['evaluation_examinations'])
            for k in range(num_of_evals):
                rct_name = json_data['plans'][i]['evaluation_examinations'][k]['name']
                
                if rct_name != 'pCTp0': # Exclude pCTp0 - WIP
                    num_of_opt_case = len(json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'])
                    for l in range(num_of_opt_case):
                        if json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'][l]['completed'] == True:
                            rct_dose_path = json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'][l]['gradient_descent_result']['dose_DICOM_filename']
                            
                            directory2, rct_dose_fn = os.path.split(rct_dose_path)
                            name2, ext2 = os.path.splitext(rct_dose_fn)
                            rct_dose_nrrd_resampled = os.path.join(data_path,directory2,name2+'_resampled.nrrd')
                            
                            final_translation_coordinate = json_data['plans'][i]['evaluation_examinations'][k]['optimization_cases'][l]['gradient_descent_result']['final_translation_coordinate']
                            final_coordinates = [final_translation_coordinate['x'],final_translation_coordinate['y'],final_translation_coordinate['z']]
                            pCT_dose_list.append(pCT_dose_nrrd_resampled)
                            rCT_dose_list.append(rct_dose_nrrd_resampled)
                            final_translation_list.append(final_coordinates)
        
        return pCT_dose_list, rCT_dose_list, final_translation_list


def create_folder_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        

def list_patient_folders(data_path):
    """
    List all directories in the base_directory.
    Each directory represents a patient.
    """
    try:
        patient_folders = [name for name in os.listdir(data_path)
                           if os.path.isdir(os.path.join(data_path, name))]
        return patient_folders
    except FileNotFoundError:
        print(f"Directory {data_path} was not found.")
        return []


def prepare_data(data_dir, patient_ids):
    pct_paths = []
    rct_paths = []
    reg_pos = []  
    plan_ids = ["P1", "P2"]

    # Load the JSON data
    with open(os.path.join(data_dir, 'file_info.json'), 'r') as json_file:
        nrrd_info = json.load(json_file)
    
    for patient_id in patient_ids:
        patient_folder = os.path.join(data_dir, patient_id)
        all_files = glob.glob(os.path.join(patient_folder, '*.nii.gz'))
        
        for plan_id in plan_ids:
            planning_ct = [f for f in all_files if f.endswith(f"{plan_id}_planningCT.nii.gz")]
            
            # Use regex to find repeated CT files that match the plan_id pattern
            pattern = re.compile(f".*{plan_id}_repeatedCT\d*\.nii.gz")
            repeated_ct_files = [f for f in all_files if pattern.match(f)]
            print(f"Repeated CT files for {plan_id}: {repeated_ct_files}")

            # Process each planning CT file
            if planning_ct and repeated_ct_files:
                # print(f"Planning CT files for {plan_id}: {planning_ct}")

                # Associate each planning CT with its corresponding repeated CTs
                for rct_path in repeated_ct_files:
                    rct_filename = os.path.basename(rct_path)
                    
                    # Look for the corresponding registration information in the JSON data
                    for patient in nrrd_info:
                        if patient['id'] == patient_id:
                            for plan_detail in patient['plan_details']:


                                for eval_exam in plan_detail['evaluation_examinations']:
                                    # print(f"Comparing {rct_filename} with {eval_exam['repeatedCT_filename']}")
                                    if rct_filename == eval_exam['repeatedCT_filename']:
                                        # print("Match found")
                                        reg_pos.append([eval_exam['final_translation_coordinate']['x'],
                                                        eval_exam['final_translation_coordinate']['y'],
                                                        eval_exam['final_translation_coordinate']['z']])
                                        rct_paths.append(rct_path)
                                        pct_paths.append(planning_ct)

                                        # Assuming unique filenames, stop searching once a match is found
                                        break
                                reg_pos_array = np.array(reg_pos, dtype=np.float32)

    return pct_paths, rct_paths, reg_pos_array


def prepare_data_nrrd(data_dir, patient_ids):
    pct_paths = []
    rct_paths = []
    reg_pos = []  
    plan_ids = ["P1", "P2"]

    # Load the JSON data
    with open(os.path.join(data_dir, 'file_info.json'), 'r') as json_file:
        nrrd_info = json.load(json_file)
        
    for patient_id in patient_ids:
        patient_folder = os.path.join(data_dir, patient_id)
        all_files = glob.glob(os.path.join(patient_folder, '*.nrrd'))
        
        for plan_id in plan_ids:
            planning_ct = [f for f in all_files if f.endswith(f"{plan_id}_planningCT.nrrd")]
            
            # Use regex to find repeated CT files that match the plan_id pattern
            pattern = re.compile(f".*{plan_id}_repeatedCT\d*\.nrrd")
            repeated_ct_files = [f for f in all_files if pattern.match(f)]
            # print(f"Repeated CT files for {plan_id}: {repeated_ct_files}")

            # Process each planning CT file
            if planning_ct and repeated_ct_files:
                # print(f"Planning CT files for {plan_id}: {planning_ct}")

                # Associate each planning CT with its corresponding repeated CTs
                for rct_path in repeated_ct_files:
                    rct_filename = os.path.basename(rct_path)
                    
                    # Look for the corresponding registration information in the JSON data
                    for patient in nrrd_info:
                        if patient['id'] == patient_id:
                            for plan_detail in patient['plan_details']:


                                for eval_exam in plan_detail['evaluation_examinations']:
                                    if rct_filename == eval_exam['repeatedCT_filename']:
                                        
                                        reg_pos.append([eval_exam['final_translation_coordinate']['x'],
                                                        eval_exam['final_translation_coordinate']['y'],
                                                        eval_exam['final_translation_coordinate']['z']])
                                        rct_paths.append(rct_path)
                                        pct_paths.append(planning_ct)
                                        # print(reg_pos)
                                        break
                                reg_pos_array = np.array(reg_pos, dtype=np.float32)

    return pct_paths, rct_paths, reg_pos_array


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate the number of samples for each set
    total_samples = len(data)
    num_train = int(train_ratio * total_samples)
    num_val = int(val_ratio * total_samples)
    num_test = total_samples - num_train - num_val
    
    # Split the data into training, validation, and test sets
    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]
    
    return train_data, val_data, test_data


def prepare_data_nrrd_for_CT(data_dir, patient_ids):
    pct_paths = []
    rct_paths = []
    reg_pos = []

    # Load the JSON data
    with open(os.path.join(data_dir, 'file_info.json'), 'r') as json_file:
        nrrd_info = json.load(json_file)
    
    for patient in nrrd_info:
        if patient['id'] in patient_ids:
            for examination_detail in patient['examination_details']:
                # Paths for planning CT and repeated CT
                planning_ct_path = os.path.join(data_dir, patient['id'], examination_detail['planningCT_filename'])
                repeated_ct_path = os.path.join(data_dir, patient['id'], examination_detail['repeatedCT_filename'])
                
                # Append paths if they exist
                if os.path.exists(planning_ct_path) and os.path.exists(repeated_ct_path):
                    pct_paths.append(planning_ct_path)
                    rct_paths.append(repeated_ct_path)
                    
                    # Append registration position
                    reg_pos.append([
                        examination_detail['final_translation_coordinate']['x'],
                        examination_detail['final_translation_coordinate']['y'],
                        examination_detail['final_translation_coordinate']['z']
                    ])
    
    reg_pos_array = np.array(reg_pos, dtype=np.float32)

    return pct_paths, rct_paths, reg_pos_array


class DataFactory:
    def __init__(self, data_path, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        self._prepare_data()

    def _prepare_data(self):
        patient_folders = [os.path.join(self.data_path, name) for name in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, name))]
        random.seed(42)  # Ensuring reproducibility
        random.shuffle(patient_folders)  # Shuffling to randomize the input data order
        
        total_patients = len(patient_folders)
        train_end = int(total_patients * self.train_ratio)
        val_end = train_end + int(total_patients * self.val_ratio)
        
        self.train_data = patient_folders[:train_end]
        self.val_data = patient_folders[train_end:val_end]
        self.test_data = patient_folders[val_end:]
        
    def get_data_split(self, split_type):
        if split_type == 'train':
            return self.train_data
        elif split_type == 'val':
            return self.val_data
        elif split_type == 'test':
            return self.test_data
        else:
            raise ValueError("Invalid data split type. Use 'train', 'val', or 'test'.")


class LoaderFactory:
    def __init__(self, data, transforms, cache_rate=0.1, num_workers=1):
        self.data = data
        self.transforms = transforms
        self.cache_rate = cache_rate
        self.num_workers = num_workers

    def get_loader(self, batch_size=1, shuffle=True):
        dataset = CacheDataset(data=self.data, transform=self.transforms, cache_rate=self.cache_rate, num_workers=self.num_workers)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.num_workers)



def save_experiment_details(data, path_experiments, 
                               train_data, val_data, test_data,
                               final_epoch, optimizer, scheduler, dim, pixdim,
                               batch_size, cache_rate, num_workers, 
                               reader, spacing_mode, spacial_pad,
                               initializer, num_filters, kernel_size, stride, dropout,
                               learning_rate, lambda_reg, weight_decay, weight_correction):
                            #df, df_loss_min, df_val_min, weight_correction):
    
    data['experiment5'] = {
        'inf_data': {
            'num_train': len(train_data),
            'percent_train': len(train_data) / (len(test_data) + len(val_data) + len(train_data)),
            'num_val': len(val_data),
            'percent_val': len(val_data) / (len(test_data) + len(val_data) + len(train_data)),
            'num_test': len(test_data),
            'percent_test': len(test_data) / (len(test_data) + len(val_data) + len(train_data))
        },
        'num_epoch': final_epoch,
        'optimizer': str(optimizer),
        'scheduler': {
            'name': str(scheduler),
            'mode': scheduler.mode,
            'patience': scheduler.patience
        },
        'dimension': dim,
        'spacing': pixdim,
        'dataloader': {
            'batch_size': batch_size,
            'cache_rate': cache_rate,
            'num_workers': num_workers
        },
        'transform': {
            'reader': str(reader),
            'spacing_mode': spacing_mode,
            'spacial_pad': spacial_pad
        },
        'model': {
            'initializer': str(initializer),
            'num_filters': num_filters,
            'num_blocks': len(num_filters),
            'kernel_size': kernel_size,
            'stride': stride,
            'dropout': dropout,
            'num_conv_block': len(kernel_size)
        },
        'learning_rate': {
            'learning_rate': learning_rate,
            'weight_correction': weight_correction,
            'lambda_reg': lambda_reg,
            'weight_decay': weight_decay
        },
        # 'best_results': {
        #     'train_loss': df.Loss.min(),
        #     'train_epoch': str(df_loss_min[0]),
        #     'val_loss': df.Val.min(),
        #     'val_epoch': str(df_val_min[0])
        # }
    }
    
    return data


from datetime import datetime

def get_date():
    """
    Returns the current date and time in the format 'day_month_hour'.
    """
    # Get the current date and time
    now = datetime.now()

    # Extract day, month, and hour
    day = now.day
    month = now.month
    hour = now.hour

    # Format as 'day_month_hour'
    formatted_date = f"{month}_{day}_{hour}"

    return formatted_date

