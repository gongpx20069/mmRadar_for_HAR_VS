from MyDataset import LmdbData
import os
import numpy as np

def parse_dir(parent_dir,sub_dir,lmdbData):
    len_max_min = [0, 1e10]
    dir=os.path.join(parent_dir,sub_dir)
    print('---------------------------------------------------')
    print('parsing dir: %s'%dir)
    label=sub_dir
    for file_name in os.listdir(dir):
        file_path=os.path.join(dir,file_name)
        print('parsing file: %s' % file_path)
        data,len_mm=parse_file(file_path)
        len_max_min = [max(len_mm[0],len_max_min[0]),min(len_mm[1],len_max_min[1])]
        for item in data:
            lmdbData.add(item,label)
    return len_max_min

def parse_file(path):
    with open(path) as f:
        lines = f.readlines()
    wordlist = []
    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)
    xs=[]
    ys=[]
    zs=[]
    ranges=[]
    velocities=[]
    doppler_bins=[]
    bearings=[]
    intensities=[]

    len_max_min=[0,1e10]
    frame_id = -1
    frame_ids = []
    for i in range(0,len(wordlist)):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_id += 1
        if wordlist[i] == "point_id:":
            frame_ids.append(frame_id)
        if wordlist[i] == "x:":
            xs.append(wordlist[i+1])
        if wordlist[i] == "y:":
            ys.append(wordlist[i+1])
        if wordlist[i] == "z:":
            zs.append(wordlist[i+1])
        if wordlist[i] == "range:":
            ranges.append(wordlist[i+1])
        if wordlist[i] == "velocity:":
            velocities.append(wordlist[i+1])
        if wordlist[i] == "doppler_bin:":
            doppler_bins.append(wordlist[i+1])
        if wordlist[i] == "bearing:":
            bearings.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensities.append(wordlist[i+1])
    sequence = [[] for i in range(frame_id+1)]  # elements are frames
    for i in range(len(xs)):
        sequence[frame_ids[i]].append([float(xs[i]),float(ys[i]),float(zs[i]),float(ranges[i]),float(velocities[i]),float(doppler_bins[i]),float(bearings[i]),float(intensities[i])])
    for frame in sequence:
        len_max_min=[max(len_max_min[0],len(frame)),min(len_max_min[1],len(frame))]
    data=[]
    window=60
    sliding=10
    frame_id=0
    while frame_id+window<len(sequence):
        data.append(sequence[frame_id:frame_id+window])
        frame_id+=sliding
    print(len_max_min)
    return data,len_max_min


parent_dir = '../Data/Train'
parent_dir2 = '../Data/Test'
sub_dirs=['boxing','jack','jump','squats','walk']


if __name__=="__main__":
    lmdbData_train=LmdbData('../Data/lmdbData_train',map_size = 2)
    lmdbData_test = LmdbData('../Data/lmdbData_test', map_size=2)
    len_max_min = [0, 1e10]
    for sub_dir in sub_dirs:
        len_mm = parse_dir(parent_dir,sub_dir,lmdbData_train)
        len_max_min = [max(len_mm[0], len_max_min[0]), min(len_mm[1], len_max_min[1])]
    for sub_dir in sub_dirs:
        len_mm = parse_dir(parent_dir2,sub_dir,lmdbData_test)
        len_max_min = [max(len_mm[0], len_max_min[0]), min(len_mm[1], len_max_min[1])]
    print(len_max_min)

