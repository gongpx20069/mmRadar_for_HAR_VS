import numpy as np
import os
from mmfall_dataset import data_preproc

def DS2run():
    # Load inference dataset and the ground truth timesheet
    inferencepath = 'DS2/DS2_bf_02'
    inferencedata = data_preproc().load_bin(inferencepath + '.npy', fortrain=False)
    # Ground truth time index file exists
    if os.path.exists(inferencepath + '.csv'): 
        gt_falls_idx = np.genfromtxt(inferencepath + '.csv', delimiter=',').astype(int)
    else:
        gt_falls_idx = []

    print(np.shape(inferencedata), gt_falls_idx)

def diff_train_test(file):
    # Load inference dataset and the ground truth timesheet
    alldata = data_preproc().load_bin(file + '.npy', fortrain=False)
    # alldata = np.load("DS2all.npy")
    # print(np.shape(alldata))
    # np.save('DS2all', alldata)
    # Ground truth time index file exists
    if os.path.exists(file + '.csv'): 
        gt_falls_idx = list(np.genfromtxt(file + '.csv', delimiter=',').astype(int))
        # print(gt_falls_idx)
    else:
        gt_falls_idx = []
    if len(gt_falls_idx) > 0:
        falls = alldata[gt_falls_idx,:]
        # print(np.shape(falls))
        normal = np.delete(alldata, gt_falls_idx, axis = 0)
    else:
        falls = None
        normal = alldata
    return falls, normal


def raw2numpy():
    files = ['DS2/DS2']
    allfalls = np.empty(shape=(0, 10, 64, 4))
    allnormal = np.empty(shape=(0, 10, 64, 4))
    for file in files:
        falls, normal = diff_train_test(file)
        allnormal = np.append(allnormal, normal, axis = 0)
        if falls is not None:
            allfalls = np.append(allfalls, falls, axis = 0)
    # print(np.shape(allfalls), np.shape(allnormal))
    np.save("DS2_falls", allfalls)
    np.save("DS2_normal", allnormal)


if __name__ == '__main__':
    raw2numpy()