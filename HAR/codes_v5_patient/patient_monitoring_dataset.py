import os
import pickle
import random

import lmdb
import rosbag
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ROS_BAG_DATA_DIR='data.ncignore/ros_bag_files'
# LMDB_DATA='data.ncignore/PMData'
ROS_BAG_DATA_DIR='../patient/ros_bag_files'
LMDB_DATA='../patient/PMData'

FRAME_LEN = 50 # ms
SEQ_LEN = 20 # a squence consists of 60 frames, that is 3000ms
SLIDE = 10
LABEL_MAP={
    'other':0,
    'walking':1,
    'falling':2,
    'swing':3,
    'sitting':4,
    'laying':5
}
class PMDataset(Dataset):
    def __init__(self,dir,map_size=1,train=True):
        self.train=train
        self.train_ratio = 0.8
        self.env = lmdb.open(dir, map_size=map_size * 1024 * 1024 * 1024)  # map_size G
        self.next = 0
        self.num_class=len(LABEL_MAP)

        txn = self.env.begin()
        # padding size = num_points_in_frame_max
        self.num_points_in_frame_max = int(txn.get('num_points_in_frame_max'.encode(),str(-1).encode()).decode())
        if self.num_points_in_frame_max==-1:
            self.num_points_in_frame_max=-1e10
        self.num_points_in_frame_min = int(txn.get('num_points_in_frame_min'.encode(), str(-1).encode()).decode())
        if self.num_points_in_frame_min == -1:
            self.num_points_in_frame_min = 1e10

        self.len = int(txn.get('len'.encode(),str(0).encode()).decode())
        #balance
        self.total_ids=[[] for _ in range(self.num_class)]
        for i in range(self.len):
            x,y=self.get(i)
            self.total_ids[y].append(i)
        num_min=min([len(ids) for ids in self.total_ids])  # number of samples in smallest class
        for i in range(len(self.total_ids)):
            random.seed(2020)
            random.shuffle(self.total_ids[i])

        self.train_len_of_each_class = int(num_min * self.train_ratio)
        self.test_len_of_each_class = (num_min-self.train_len_of_each_class)

        self.test_len=self.test_len_of_each_class*self.num_class
        self.train_len=self.train_len_of_each_class*self.num_class

        # generate ids of testset
        self.test_ids = []
        for i in range(self.num_class):
            self.test_ids.extend(self.total_ids[i][:self.test_len_of_each_class])
        random.seed(2020)
        random.shuffle(self.test_ids)

        # get all trainset ids
        self.total_ids_train = [[] for _ in range(self.num_class)]
        for i in range(self.num_class):
            self.total_ids_train[i] = self.total_ids[i][self.test_len_of_each_class:]
        # generate ids of trainset
        self.train_ids = []
        for i in range(self.num_class):
            random.seed(2020)
            random.shuffle(self.total_ids_train[i])
            self.train_ids.extend(self.total_ids_train[i][:self.train_len_of_each_class])
        random.seed(2020)
        random.shuffle(self.train_ids)

        # self.train_len=int(self.len*0.8)
        # self.test_len = self.len-int(self.len * self.train_ratio)
        #
        # self.ids=[i for i in range(self.len)]
        # random.seed(2020)
        # random.shuffle(self.ids)
        # self.train_ids=self.ids[:self.train_len]
        # self.test_ids = self.ids[self.train_len:]

    def shuffle(self):
        self.train_ids=[]
        for i in range(self.num_class):
            random.shuffle(self.total_ids_train[i])
            self.train_ids.extend(self.total_ids_train[i][:self.train_len_of_each_class])
        random.shuffle(self.train_ids)

    def __len__(self):
        if self.train:
            return self.train_len
        else:
            return self.test_len

    def __getitem__(self, item):
        # x shape(SEQ_LEN, FRAME_LEN, Feature)
        if self.train:
            x,y=self.get(self.train_ids[item])
        else:
            x, y = self.get(self.test_ids[item])

        # zero padding
        input=np.zeros((len(x),self.num_points_in_frame_max,len(x[0][0])))
        for i,frame in enumerate(x):
            input[i,:len(frame)]=np.array(frame)
        input_xyz=input[:,:,:3]
        input_xyz_tensor = torch.FloatTensor(input_xyz)
        input_tensor = torch.FloatTensor(np.delete(input, 9, axis=2))
        label_tensor = torch.tensor(y).long()

        input_xyz_tensor = torch.where(torch.isnan(input_xyz_tensor), torch.full_like(input_xyz_tensor, 0), input_xyz_tensor)
        input_tensor = torch.where(torch.isnan(input_tensor), torch.full_like(input_tensor, 0),input_tensor)

        a = torch.isnan(input_xyz_tensor).sum()
        b = torch.isnan(input_tensor).sum()
        if a > 0 or b > 0:
            print('error: input data nan, exit')
            exit(-1)
        return input_xyz_tensor,input_tensor,label_tensor

    def get(self, id):
        txn = self.env.begin()
        x = pickle.loads(txn.get(('x_%d' % (id)).encode()))
        y = int(txn.get(('y_%d' % (id)).encode()).decode())
        return x, y

    def add(self, x, y):
        # x shape(SEQ_LEN, FRAME_LEN, Feature)
        # Feature   0,  1,  2,  3,      4,      5,      6,      7,      8,          9,              10,         11
        #           x,  y,  z,  posX,   posY,   range,  velX    velY,   velocity,   timestamp(ns)   bearing,    doppler_bin
        txn = self.env.begin(write=True)
        txn.put(('x_%d' % (self.next)).encode(), pickle.dumps(x))
        txn.put(('y_%d' % (self.next)).encode(), str(y).encode())
        txn.put('len'.encode(), str(self.next + 1).encode())
        self.num_points_in_frame_max=max(self.num_points_in_frame_max, max([len(frame) for frame in x]))
        self.num_points_in_frame_min = min(self.num_points_in_frame_min, min([len(frame) for frame in x]))
        txn.put('num_points_in_frame_max'.encode(), str(self.num_points_in_frame_max).encode())
        txn.put('num_points_in_frame_min'.encode(), str(self.num_points_in_frame_min).encode())
        txn.commit()
        self.next += 1

    def put(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k, v)
        txn.commit()



def read_bag(file):
    print('read file:', file)
    data = []
    with rosbag.Bag(file) as bag:
        bag_data = bag.read_messages('/ti_mmwave/radar_scan')
        frame=[]
        frame_start_time=-1
        for topic, msg, t in bag_data:
            t_ns=int(str(t))
            try:
                feature=[msg.x, msg.y,msg.z,
                         msg.posX,msg.posY,msg.range,
                         msg.velX,msg.vely,msg.velocity,
                         t_ns,msg.bearing,msg.doppler_bin]
            except:
                try:
                    feature = [msg.x, msg.y, msg.z,
                               msg.posX, msg.posY, msg.range,
                               msg.velX, msg.velY, msg.velocity,
                               t_ns, msg.bearing, msg.doppler_bin]
                except:
                    feature = [msg.x, msg.y, msg.z,
                               0, 0, msg.range,
                               0, 0, msg.velocity,
                               t_ns, msg.bearing, msg.doppler_bin]
            if frame_start_time==-1:
                frame_start_time=t_ns
            if t_ns-frame_start_time > FRAME_LEN*1e6:
                data.append(frame)
                frame=[feature]
                frame_start_time = t_ns
            else:
                frame.append(feature)
            # if np.isnan(np.array(feature)).any():
            #     print('error: nan')
            #     print(msg)
            #     print(feature)
            #     exit(-1)
    return data

def bags_to_lmdb(dir):
    lmdbData=PMDataset(dir=LMDB_DATA,map_size=10)
    for file in os.listdir(dir):
        label=None
        for key in LABEL_MAP:
            if file.find(key)!=-1:
                label=key
                break
        path=os.path.join(dir,file)
        if label is None:
            print('wrong label in file:', path)
            exit(-1)
        frames = read_bag(path)
        start=0
        while start+SEQ_LEN<len(frames):
            seq=frames[start:start+SEQ_LEN]
            label_id=LABEL_MAP[label]
            lmdbData.add(seq,label_id)
            start+=SLIDE
    print('write lmdb data, done.')

if __name__=='__main__':
    # bags_to_lmdb(ROS_BAG_DATA_DIR)

    train_data=PMDataset(dir=LMDB_DATA,map_size=10,train=True)
    test_data = PMDataset(dir=LMDB_DATA, map_size=10, train=False)

    print('train dataset length:', train_data.__len__())
    print('test dataset length:', test_data.__len__())
    print(train_data.num_points_in_frame_max,train_data.num_points_in_frame_min)
    print('read dataset testing...')
    cnt=np.zeros(len(LABEL_MAP))
    for i in tqdm(range(train_data.__len__())):
        input_xyz, input, label=train_data.__getitem__(i)
        cnt[label.item()]+=1
        pass
    print('done~!')
    print(cnt)

