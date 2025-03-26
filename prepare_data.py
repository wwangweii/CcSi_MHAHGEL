import numpy as np
import pandas as pd
import os
import sys
import scipy.io as io

dir = "put/your/path/to/prepared/dataset/ABIDE/"
dir = "/home/proj-openset/dataset/ABIDE/"
filename = "put/your/path/to/dataset/ABIDE/Phenotypic_V1_0b_preprocessed1.csv"
filename = "/home/proj-openset/dataset/ABIDE/Phenotypic_V1_0b_preprocessed1.csv"
data = pd.read_csv(filename, header=0, index_col=0)
dataframe = pd.DataFrame(data, columns=None)
header = np.array(dataframe.columns)
try:
    site_idx = dataframe['SITE_ID']
    file_idx = dataframe['FILE_ID']
    age_idx = dataframe['AGE_AT_SCAN']
    sex_idx = dataframe['SEX']
    dx_idx = dataframe['DX_GROUP']
    mean_fd_idx = dataframe['func_mean_fd']
except Exception as exc:
    err_msg = 'Unable to extract header information from the pheno file: {0}\nHeader should have pheno info:' \
              ' {1}\nError: {2}'.format(filename, str(header), exc)
    raise Exception(err_msg)

print(site_idx)
table = pd.concat([file_idx, site_idx, age_idx, sex_idx, dx_idx, mean_fd_idx], axis=1)
table.set_index('FILE_ID', inplace=True)
print(table)


file_lookup = {"aal": "rois_aal dparsf filt_global",
               "ho": "rois_ho dparsf filt_global",
               "dosenbach160": "rois_dosenbach160 dparsf filt_global"}

roi_lookup = {"aal":116,
               "ho": 111,
               "dosenbach160": 161}


def load_data(*input_atlas_list):
    if len(input_atlas_list)==3:
        return load_pipline_with_three_atlas()
    elif len(input_atlas_list)==2:
        return load_pipline_with_two_atlas(*input_atlas_list)
    elif len(input_atlas_list)==1:
        return load_pipline_with_one_atlas(*input_atlas_list)
    else:
        raise OSError("only support maximum 3 atlas")
    
def load_pipline_with_three_atlas():
    path1 = "rois_aal dparsf filt_global"
    path2 = "rois_ho dparsf filt_global"
    path3 = "rois_dosenbach160 dparsf filt_global"
    all_atlas = ['aal', 'ho', 'dosenbach160']
    files1 = os.listdir(dir + path1) #得到文件夹下的所有文件名称
    files2 = os.listdir(dir + path2)
    files3 = os.listdir(dir + path3)
    sub1 = len(files1)
    sub2 = len(files2)
    sub3 = len(files3)
    ROI1 = 116
    ROI2 = 111
    ROI3 = 161
    print("The Total Subject of AAL: ", sub1)
    print("The Total Subject of HO: ", sub2)
    print("The Total Subject of DO160: ", sub3)
    site = []
    age = []
    sex = []
    diagnosis = []
    mean_fd = []
    id1 = []
    id2 = []
    id3 = []
    for i, file in enumerate(files1):
        id1.append(file[0:-10-len(all_atlas[0])])
    for i, file in enumerate(files2):
        id2.append(file[0:-10-len(all_atlas[1])])
    for i, file in enumerate(files3):
        id3.append(file[0:-10-len(all_atlas[2])])
    inter_id = list((set(id1) & set(id2)) & set(id3))
    inter_id.sort()
    print(inter_id)
    print("The Total Inter Subject: ",len(inter_id))
    mat1 = np.zeros((len(inter_id), ROI1 * ROI1))
    mat2 = np.zeros((len(inter_id), ROI2 * ROI2))
    mat3 = np.zeros((len(inter_id), ROI3 * ROI3))
    for i, index in enumerate(inter_id):
        temp = table.loc[index]
        site.append(temp.SITE_ID)
        sex.append(temp.SEX)
        age.append(temp.AGE_AT_SCAN)
        diagnosis.append(temp.DX_GROUP)
        mean_fd.append(temp.func_mean_fd)
        mat1[i, :] = np.reshape(np.corrcoef(np.transpose(io.loadmat(dir + path1 + "/" + index + "_rois_aal.mat")['A'])) - np.eye(ROI1), (1, ROI1 * ROI1))
        mat2[i, :] = np.reshape(np.corrcoef(np.transpose(io.loadmat(dir + path2 + "/" + index + "_rois_ho.mat")['A'])) - np.eye(ROI2), (1, ROI2 * ROI2))
        mat3[i, :] = np.reshape(np.corrcoef(np.transpose(io.loadmat(dir + path3 + "/" + index + "_rois_dosenbach160.mat")['A'])) - np.eye(ROI3), (1, ROI3 * ROI3))
    bool1 = np.isnan(mat1).any(axis=1)
    bool2 = np.isnan(mat2).any(axis=1)
    bool3 = np.isnan(mat3).any(axis=1)
    bool = (~bool1) & (~bool2) & (~bool3)
    data1 = np.array(mat1[bool, :])
    data2 = np.array(mat2[bool, :])
    data3 = np.array(mat3[bool, :])
    df = pd.DataFrame({'site':site,'diagnosis':diagnosis,'age': age,'sex':sex,'mean_fd':mean_fd}, index=None)
    information = df[bool].reset_index()
    return data1, data2, data3, information


def load_pipline_with_two_atlas(atlas1, atlas2):
    
    path1 = file_lookup[atlas1]
    path2 = file_lookup[atlas2]
    files1 = os.listdir(dir + path1) #得到文件夹下的所有文件名称
    files2 = os.listdir(dir + path2)
    sub1 = len(files1)
    sub2 = len(files2)
    ROI1 = roi_lookup[atlas1]
    ROI2 = roi_lookup[atlas2]
    print(f"The Total Subject of {atlas1}: ", sub1)
    print(f"The Total Subject of {atlas2}: ", sub2)
    site = []
    age = []
    sex = []
    diagnosis = []
    mean_fd = []
    id1 = []
    id2 = []
    for i, file in enumerate(files1):
        id1.append(file[0:-10-len(atlas1)])
    for i, file in enumerate(files2):
        id2.append(file[0:-10-len(atlas2)])
    inter_id = list(set(id1) & set(id2))
    inter_id.sort()
    print("The Total Inter Subject: ",len(inter_id))
    mat1 = np.zeros((len(inter_id), ROI1 * ROI1))
    mat2 = np.zeros((len(inter_id), ROI2 * ROI2))
    for i, index in enumerate(inter_id):

        temp = table.loc[index]
        site.append(temp.SITE_ID)
        age.append(temp.AGE_AT_SCAN)
        sex.append(temp.SEX)
        diagnosis.append(temp.DX_GROUP)
        mean_fd.append(temp.func_mean_fd)
        mat1[i, :] = np.reshape(np.corrcoef(np.transpose(io.loadmat(dir + path1 + "/" + index + f"_rois_{atlas1}.mat")['A'])) - np.eye(ROI1), (1,ROI1 * ROI1))
        mat2[i, :] = np.reshape(np.corrcoef(np.transpose(io.loadmat(dir + path2 + "/" + index + f"_rois_{atlas2}.mat")['A'])) - np.eye(ROI2), (1,ROI2 * ROI2))
    bool1 = np.isnan(mat1).any(axis=1)
    bool2 = np.isnan(mat2).any(axis=1)
    bool = (~bool1) & (~bool2)
    data1 = np.array(mat1[bool, :])
    data2 = np.array(mat2[bool, :])
    df = pd.DataFrame({'site':site,'diagnosis':diagnosis,'age': age,'sex':sex,'mean_fd':mean_fd}, index=None)
    information = df[bool].reset_index()
    return data1, data2, information

def load_pipline_with_one_atlas(atlas):
    path = file_lookup[atlas]
    files = os.listdir(dir + path)
    sub = len(files)
    ROI = roi_lookup[atlas]
    print("The Total Subject: ", sub)
    site = []
    age = []
    sex = []
    diagnosis = []
    mean_fd = []
    mat = np.zeros((sub,ROI*ROI))
    for i, file in enumerate(files):
        id = file[0:-10-len(atlas)]
        temp = table.loc[id]
        site.append(temp.SITE_ID)
        age.append(temp.AGE_AT_SCAN)
        sex.append(temp.SEX)
        diagnosis.append(temp.DX_GROUP)
        mean_fd.append(temp.func_mean_fd)
        mat[i,:] = np.reshape(np.corrcoef(np.transpose(io.loadmat(dir + path + "/" + file)['A'])) - np.eye(ROI), (1,ROI*ROI))
    new_mat = mat[~np.isnan(mat).any(axis=1)]
    print("The Number of Processed Subject: ", np.size(new_mat,0))
    data = np.array(new_mat)
    df = pd.DataFrame({'site':site,'diagnosis':diagnosis,'age': age,'sex':sex,'mean_fd':mean_fd}, index=None)
    new_df = df[~np.isnan(mat).any(axis=1)].reset_index()
    return data, new_df



if __name__ == "__main__":
    pass