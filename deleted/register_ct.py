




import os
import copy
import numpy as np
from CT import construct_CT_object


def get_path_structures_DICOM_file(path_CT_DICOM_series):
    filenames = [filename for filename in os.listdir(path_CT_DICOM_series) if filename.startswith('RS')]
    print(filenames)

    assert len(filenames) > 0, "no structures DICOM file found"
    assert len(filenames) <= 1, "more than one structures DICOM files found"

    return os.path.join(path_CT_DICOM_series, filenames[0])



root_patient = "/data/oosterhoff/patients/DBP_OP007/"
name_fixed_CT = 'pCTp0'
name_moving_CT = 'rCTp12'

path_fixed_CT = os.path.join(root_patient, name_fixed_CT, '')
path_moving_CT = os.path.join(root_patient, name_moving_CT, '')


path_struct_fixed_CT = get_path_structures_DICOM_file(path_fixed_CT)
path_struct_moving_CT = get_path_structures_DICOM_file(path_moving_CT)



fixed_CT = construct_CT_object(name_fixed_CT, path_fixed_CT, path_struct_fixed_CT, roi_names = ['External'])
moving_CT = construct_CT_object(name_moving_CT, path_moving_CT, path_struct_moving_CT, roi_names = ['External'])


# transformation_matrix = [
#                         0.9998813462333056,
#                         0.015269076566525777,
#                         -0.002036849399907398,
#                         0.4487213396786764,
#                         -0.01531200129301428,
#                         0.999616966465917,
#                         -0.023053480645674963,
#                         -11.862892854566496,
#                         0.0016840638571796252,
#                         0.023081933504005992,
#                         0.9997321582677241,
#                         0.1085704794000017,
#                         0,
#                         0,
#                         0,
#                         1
#                     ]




transformation_matrix = copy.deepcopy(original_FoR_registration['rigid_transformation_matrix'])
for idx in [3, 7, 11]:
    transformation_matrix[idx] *= 10 # to go from cm to mm
transformation_matrix = np.array(transformation_matrix).reshape(4,4)
if exam_name == original_FoR_registration['to_examination_name']:
    transformation_matrix = np.linalg.inv(transformation_matrix)
moving_CT.transform(transformation_matrix, fixed_CT)


fixed_CT.override_air_outside_external()
moving_CT.override_air_outside_external()

fixed_CT.save("", save_struct_file = False)
moving_CT.save("", save_struct_file = False)



#################################################################



import matplotlib.pyplot as plt
import SimpleITK as sitk
fixed_CT_array = sitk.GetArrayFromImage(fixed_CT.image)
moving_CT_array = sitk.GetArrayFromImage(moving_CT.image)
plt.imshow((moving_CT_array - fixed_CT_array)[70])


###################################################################################################



for exam_name, exam in case_data['examinations'].items():
    if len(exam['FoR_registrations']):
        original_FoR_registration = case_data['examinations'][exam_name]['FoR_registrations'][0]
        if type(original_FoR_registration['from_examination_name']) == list:
            original_FoR_registration['from_examination_name'] = exam_name
        if type(original_FoR_registration['to_examination_name']) == list:
            if plan['planning_examination']['name'] in original_FoR_registration['to_examination_name']:
                original_FoR_registration['to_examination_name'] = name_fixed_CT
            else:
                original_FoR_registration['to_examination_name'] = original_FoR_registration['to_examination_name'][0]
    elif case_data['examinations'][name_fixed_CT]['FoR_registrations'][0]['to_examination_name'] == exam_name:
        original_FoR_registration = case_data['examinations'][name_fixed_CT]['FoR_registrations'][0]
        if type(original_FoR_registration['to_examination_name']) == list:
            original_FoR_registration['from_examination_name'] = exam_name
        if type(original_FoR_registration['from_examination_name']) == list:
            original_FoR_registration['to_examination_name'] = name_fixed_CT


###########################################
import random            
for i in range(number_of_new_cases):
    translations = {axis: (2 * random.random() - 1) * max_translation if axis in ['x', 'y', 'z'] else 0 for axis in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']}

    opt_case = copy.deepcopy(original_opt_case)

    opt_case['initial_coordinate']['x'] = translations['x']
    opt_case['initial_coordinate']['y'] = translations['y']
    opt_case['initial_coordinate']['z'] = translations['z']