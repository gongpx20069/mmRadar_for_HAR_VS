from torch.utils.data import Dataset
import numpy as np
import torch
import os

class data_preproc:
    def __init__(self):
        # One motion pattern is as (self.frames_per_pattern, self.points_per_frame, self.features_per_point)
        self.frames_per_pattern     = 10 # For 1 second for 10 fps radar rate
        self.points_per_frame       = 64 # We want to oversample every radar frame to 64 points while keeping the mean and variance the same
        self.features_per_point     = 4  # The radar can provides us (x, y, z, Doppler, RCS), but we only keep the first four feature, i.e. (x, y, z, Doppler)

        # Train and test data split ratio
        self.split_ratio            = 0.8

        # Rotation matrix due to tilt angle
        tilt_angle  = 10.0 # degrees
        self.height = 1.8   # meters
        self.rotation_matrix = np.array([[1.0, 0.0, 0.0],\
                                        [0.0, np.cos(np.deg2rad(tilt_angle)), np.sin(np.deg2rad(tilt_angle))],\
                                        [0.0, -np.sin(np.deg2rad(tilt_angle)), np.cos(np.deg2rad(tilt_angle))]])
             
    def load_bin(self, binfile_path, fortrain=True):
        # Record centroid history for analysis purpose
        centroidX_his = []
        centroidY_his = []
        centroidZ_his = []

        # Load the recorded bin file
        raw_pointcloud = np.load(binfile_path, allow_pickle=True)

        # Accumulated the motion patterns with (self.frames_per_pattern) frames
        total_pattern = []
        for idx in range(len(raw_pointcloud)-self.frames_per_pattern):
            total_pattern.append(raw_pointcloud[idx : (idx + self.frames_per_pattern)])

        # Original point vector in the .bin file:
        #   [frame number, point ID, target ID, \\ Basic information
        #   centroidX, centroidY, centroidZ, centroidVelocityX, centroidVelocityY, centroidVelocityZ, \\ Centorid information
        #   range, azimuth angle, elevation angle, Doppler, SNR, noise level] \\ Point information
        # Extract the feature vector (delta_x, delta_y, z, D, pointRCS) from the original point vector and do data oversampling proposed in the paper
        total_processed_pattern = []
        total_processed_no_norm = []
        for pattern in total_pattern:
            # Get the centroid information from the very first frame in a pattern and do coordiante transformation
            # As the centroid is already in the radar Cartesian coordinates, we just need to transfer it to the ground Cartesian coordinates
            centroidX, centroidY, centroidZ, centroidVx, centroidVy, centroidVz = pattern[0][0][3], pattern[0][0][4], pattern[0][0][5], pattern[0][0][6], pattern[0][0][7], pattern[0][0][8]
            results      = np.matmul(self.rotation_matrix, np.array([centroidX,centroidY,centroidZ]))
            centroidX    = results[0]
            centroidY    = results[1]
            centroidZ    = results[2] + self.height

            # Record the centroid history over time
            centroidX_his.append(centroidX)
            centroidY_his.append(centroidY)
            centroidZ_his.append(centroidZ)

            processed_pattern  = []
            for frame in pattern:
                processed_frame = []
                for point in frame:        
                    # Get the original point information.
                    pointR, pointAZ, pointEL, pointD, pointSNR, pointNoise = point[9], point[10], point[11], point[12], point[13], point[14]

                    # Get the point's position in the Cartesian coord and then do coordiante transformation
                    pointX      = pointR*np.cos(pointEL)*np.sin(pointAZ)
                    pointY      = pointR*np.cos(pointEL)*np.cos(pointAZ)
                    pointZ      = pointR*np.sin(pointEL)
                    results     = np.matmul(self.rotation_matrix, np.array([pointX, pointY, pointZ]))
                    pointX      = results[0]
                    pointY      = results[1]
                    pointZ      = results[2] + self.height
                    
                    # Subtract the point's position from the centroid in the very first frame in a motion pattern
                    delta_x     = pointX - centroidX
                    delta_y     = pointY - centroidY
                    delta_z     = pointZ
                    delta_D     = pointD
                    pointRCS    = 4*10*np.log10(pointR) + pointSNR*0.1 + pointNoise*0.1 # in dBsm

                    # Form the feature vector for each frame
                    feature_vector = [delta_x, delta_y, delta_z, delta_D, pointRCS]
                    processed_frame.append(feature_vector[0:self.features_per_point]) # Only keep 3D spatial info and the Doppler
                processed_pattern.append(processed_frame)
                # Do the data oversampling proposed in the paper
                processed_pattern_oversampled = self.proposed_oversampling(processed_pattern)
            total_processed_pattern.append(processed_pattern_oversampled)

        total_processed_pattern_np = np.array(total_processed_pattern)
        
        # Train and test split
        split_idx   = int(total_processed_pattern_np.shape[0]*self.split_ratio)
        traindata   = total_processed_pattern_np[0:split_idx]
        testdata    = total_processed_pattern_np[split_idx:]

        if fortrain is True: # For training, need data split to obtain both training and testing dataset
            print("INFO: Total normal motion pattern data shape: " + str(total_processed_pattern_np.shape))
            print("INFO: Training motion pattern data shape" + str(traindata.shape))
            print("INFO: Testing motion pattern data shape" + str(testdata.shape))
            return traindata, testdata
        else: # For inference on anomaly dataset
            print("INFO: Total inference motion pattern data shape: " + str(total_processed_pattern_np.shape))
            # return total_processed_pattern, centroidZ_his
            return total_processed_pattern

    def proposed_oversampling(self, processed_pointcloud):
        # # Check the input
        # point_list = []
        # for frame in processed_pointcloud:
        #     point_list.extend(frame)
        # point_list_np  = np.array(point_list)
        # assert (point_list_np.shape[-1] == self.features_per_point), ("ERROR: Input processed_pointcloud has different feature length rather than %s!" %(self.features_per_point))

        # Do the data oversampling
        processed_pointcloud_oversampled = []
        for frame in processed_pointcloud:
            frame_np = np.array(frame)
            # Check if it's empty frame
            N = self.points_per_frame
            M = frame_np.shape[0]
            assert (M != 0), "ERROR: empty frame detected!"
            # Rescale and padding
            mean        = np.mean(frame_np, axis=0)
            sigma       = np.std(frame_np, axis=0)
            frame_np    = np.sqrt(N/M)*frame_np + mean - np.sqrt(N/M)*mean # Rescale
            frame_oversampled = frame_np.tolist()
            frame_oversampled.extend([mean]*(N-M)) # Padding with mean
            # # Check if mean and sigma keeps the same. Comment for saving time.
            # new_mean    = np.mean(np.array(frame_oversampled), axis=0)
            # new_sigma   = np.std(np.array(frame_oversampled), axis=0)
            # assert np.sum(np.abs(new_mean-mean))<1e-5, ("ERROR: Mean rescale and padding error!")
            # assert np.sum(np.abs(new_sigma-sigma))<1e-5, ("ERROR: Sigma rescale and padding error!")
            processed_pointcloud_oversampled.append(frame_oversampled)

        processed_pointcloud_oversampled_np = np.array(processed_pointcloud_oversampled)
        assert (processed_pointcloud_oversampled_np.shape[-2] == self.points_per_frame), ("ERROR: The new_frame_data has different number of points per frame rather than %s!" %(self.points_per_frame))    
        assert (processed_pointcloud_oversampled_np.shape[-1] == self.features_per_point), ("ERROR: The new_frame_data has different feature length rather than %s!" %(self.features_per_point))    

        return processed_pointcloud_oversampled_np

class Falldataset(Dataset):
    def __init__(self, root_falls = "./DS1/DS1_4falls", root_normal = "./DS1/DS1_4normal" , raw_data = True):
        self.root_falls = root_falls
        self.root_normal = root_normal

        if raw_data:
            self.falls = data_preproc().load_bin(self.root_falls + '.npy', fortrain=False)
            self.normal = data_preproc().load_bin(self.root_normal + '.npy', fortrain=False)
        else:
            self.falls = np.load(self.root_falls + '.npy')
            self.normal = np.load(self.root_normal + '.npy')

        arr = np.arange(self.__len__())
        np.random.shuffle(arr)
        self.arr = arr

    def replace_raw(self):
        np.save(self.root_falls,self.falls)
        np.save(self.root_normal, self.normal)
        print("save them in falls: {} and normal:{}".format(self.root_falls, self.root_normal))

    def __len__(self):
        return np.shape(self.falls)[0] + np.shape(self.normal)[0]

    def __getitem__(self,index):
        if index < np.shape(self.falls)[0]:
            x = self.falls[index]
            y = 1
        else:
            index = index - np.shape(self.falls)[0]
            x = self.normal[index]
            y = 0

        return torch.tensor(x[:,:,0:3]).float(),torch.tensor(x).float(), torch.tensor(y).long()

if __name__ == '__main__':
    dataset = Falldataset()
    a,b,c = dataset[2000]
    print(a.size(),b.size().c.size())
