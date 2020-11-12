
from MyDataset import LmdbData

parent_dir = './Data/Train'
parent_dir2 = './Data/Test'
sub_dirs=['boxing','jack','jump','squats','walk']



import glob
import os
import numpy as np
import csv
import time
import time




def voxalize(x_points, y_points, z_points, x, y, z, velocity):
    x_min = np.min(x)
    x_max = np.max(x)

    y_min = np.min(y)
    y_max = np.max(y)

    z_max = np.max(z)
    z_min = np.min(z)

    z_res = (z_max - z_min)/z_points
    y_res = (y_max - y_min)/y_points
    x_res = (x_max - x_min)/x_points

    pixel = np.zeros([x_points,y_points,z_points])

    for i in range(y.shape[0]):
        x_current = x_min
        x_prev = x_min
        x_count = 0
        done=False

        while x_current <= x_max and x_count < x_points and done==False:
            y_prev = y_min
            y_current = y_min
            y_count = 0
            while y_current <= y_max and y_count < y_points and done==False:
                z_prev = z_min
                z_current = z_min
                z_count = 0
                while z_current <= z_max and z_count < z_points and done==False:
                    if x[i] < x_current and y[i] < y_current and z[i] < z_current and x[i] >= x_prev and y[i] >= y_prev and z[i] >= z_prev:
                        pixel[x_count,y_count,z_count] = pixel[x_count,y_count,z_count] + 1
                        done = True

                        #velocity_voxel[x_count,y_count,z_count] = velocity_voxel[x_count,y_count,z_count] + velocity[i]
                    z_prev = z_current
                    z_current = z_current + z_res
                    z_count = z_count + 1
                y_prev = y_current
                y_current = y_current + y_res
                y_count = y_count + 1
            x_prev = x_current
            x_current = x_current + x_res
            x_count = x_count + 1
    return pixel


def get_data(file_path,label,lmdbData):
    with open(file_path) as f:
        lines = f.readlines()

    frame_num_count = -1
    frame_num = []
    x = []
    y = []
    z = []
    velocity = []
    intensity = []
    wordlist = []


    for x1 in lines:
        for word in x1.split():
            wordlist.append(word)

    length1 = len(wordlist)

    for i in range(0,length1):
        if wordlist[i] == "point_id:" and wordlist[i+1] == "0":
            frame_num_count += 1
        if wordlist[i] == "point_id:":
            frame_num.append(frame_num_count)
        if wordlist[i] == "x:":
            x.append(wordlist[i+1])
        if wordlist[i] == "y:":
            y.append(wordlist[i+1])
        if wordlist[i] == "z:":
            z.append(wordlist[i+1])
        if wordlist[i] == "velocity:":
            velocity.append(wordlist[i+1])
        if wordlist[i] == "intensity:":
            intensity.append(wordlist[i+1])

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    frame_num = np.asarray(frame_num)
    velocity = np.asarray(velocity)
    intensity = np.asarray(intensity)

    x = x.astype(np.float)
    y = y.astype(np.float)
    z = z.astype(np.float)
    velocity = velocity.astype(np.float)
    intensity = intensity.astype(np.float)
    frame_num = frame_num.astype(np.int)


    data = dict()

    for i in range(len(frame_num)):
        if int(frame_num[i]) in data:
            data[frame_num[i]].append([x[i],y[i],z[i],velocity[i],intensity[i]])
        else:
            data[frame_num[i]]=[]
            data[frame_num[i]].append([x[i],y[i],z[i],velocity[i],intensity[i]])

    data_pro1 = dict()

    # Merging of frames together with sliding of  frames
    together_frames = 1
    sliding_frames = 1

    #we have frames in data
    frames_number = []
    for i in data:
        frames_number.append(i)

    frames_number=np.array(frames_number)
    total_frames = frames_number.max()

    i = 0
    j = 0

    while together_frames+i < total_frames:

        curr_j_data =[]
        for k in range(together_frames):
            curr_j_data = curr_j_data + data[i+k]
        #print(len(curr_j_data))
        data_pro1[j] = curr_j_data
        j = j+1
        i = i+sliding_frames

    pixels = []

    # Now for 2 second windows, we need to club together the frames and we will have some sliding windows
    for i in data_pro1:
        f = data_pro1[i]
        f = np.array(f)

        #y and z points in this cluster of frames
        x_c = f[:,0]
        y_c = f[:,1]
        z_c = f[:,2]
        vel_c=f[:,3]

        pix = voxalize(10, 32, 32, x_c, y_c, z_c, vel_c)
        #print(i, f.shape,pix.shape)
        pixels.append(pix)

    pixels = np.array(pixels)

    frames_together = 60
    sliding = 10

    i = 0
    while i+frames_together<=pixels.shape[0]:
        local_data=[]
        for j in range(frames_together):
            local_data.append(pixels[i+j])

        local_data=np.array(local_data)
        lmdbData.add(local_data,label)
        i = i + sliding

# parse the data file


def parse_RF_files(parent_dir, sub_dir, lmdbData, file_ext='*.txt'):
    print(sub_dir)
    files=sorted(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)))
    for fn in files:
        print(sub_dir,fn)
        get_data(fn,sub_dir,lmdbData)


if __name__=="__main__":
    lmdbData=LmdbData('./Data/lmdbData_train',map_size = 60)
    for sub_dir in sub_dirs:
        parse_RF_files(parent_dir,sub_dir,lmdbData)

    lmdbData=LmdbData('./Data/lmdbData_test', map_size = 30)
    for sub_dir in sub_dirs:
        parse_RF_files(parent_dir2,sub_dir,lmdbData)

