import os
import pickle
import random
import numpy as np
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

LMDB_DATASET_ROOT='../gesture_dataset/lmdbData'
DATA_SOURCE='../gesture_dataset/long_SEP'
SEQ_LEN=20
SEQ_LEN_DROP=5
SLIDE_WINDOW=5
LABEL_MAP={
    'knock':0,
    'rotate':1,
    'lswipe':2,
    'rswipe':3,
    # 'run':4,
    # 'other':5
}
NUM_CLASS=len(LABEL_MAP)

class GestureDataset(Dataset):
    def __init__(self,dir=LMDB_DATASET_ROOT,map_size=1,train=True, with_state=False, normlize='max-min'): # nomalize = 'max-min'
        self.train=train
        self.with_state=with_state # if set false, returns only xyz features
        self.normlize=normlize
        self.train_ratio = 0.8
        self.env = lmdb.open(dir, map_size=map_size * 1024 * 1024 * 1024)  # map_size G
        self.next = 0

        txn = self.env.begin()
        # padding size = num_points_in_frame_max
        self.num_points_in_frame_max = int(txn.get('num_points_in_frame_max'.encode(),str(-1).encode()).decode())
        if self.num_points_in_frame_max==-1:
            self.num_points_in_frame_max=-1e10
        self.num_points_in_frame_min = int(txn.get('num_points_in_frame_min'.encode(), str(-1).encode()).decode())
        if self.num_points_in_frame_min == -1:
            self.num_points_in_frame_min = 1e10

        self.len = int(txn.get('len'.encode(),str(0).encode()).decode())
        self.train_len=int(self.len*0.8)
        self.test_len = self.len-int(self.len * self.train_ratio)

        self.ids=[i for i in range(self.len)]
        random.seed(2020)
        random.shuffle(self.ids)
        self.train_ids=self.ids[:self.train_len]
        self.test_ids = self.ids[self.train_len:]

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
        #compute max min
        if self.normlize=='max-min':
            max5,min5=np.zeros(len(x[0][0]))-1e10,np.zeros(len(x[0][0]))+1e10
            for i, frame in enumerate(x):
                frame = np.array(frame)
                max_i,min_i=frame.max(0), frame.min(0)
                max5,min5=np.max((max5,max_i),axis=0),np.min((min5,min_i),axis=0)
        # zero padding
        input=np.zeros((SEQ_LEN,self.num_points_in_frame_max,len(x[0][0])))
        for i,frame in enumerate(x):
            frame=np.array(frame)
            if self.normlize == 'max-min':
                frame = (frame - min5) / (max5 - min5)  # max-min
            if np.isnan(frame).any():
                print('nan:',item)
                exit(-1)
            input[i,:len(frame)]=frame

        if not self.with_state:
            input = input[:, :, :3] # only x,y,z

        input = torch.tensor(input).float()
        label = torch.tensor(y).long()
        return input[:, :, :3],input,label

    def get(self, id):
        txn = self.env.begin()
        x = pickle.loads(txn.get(('x_%d' % (id)).encode()))
        y = int(txn.get(('y_%d' % (id)).encode()).decode())
        return x, y

    def add(self, x, y):
        # x shape(SEQ_LEN, FRAME_LEN, Feature)
        # Feature   0,  1,  2,  3,      4,
        #           x,  y,  z,  doppler,intensity,
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

def read_file(path):
    print('\treading file: %s' % (path))
    with open(path,'r') as f:
        lines=f.readlines()
        data_origin=[]
        data=[]
        for i, line in enumerate(lines):
            if i==0:
                continue
            data_line=list(map(eval,line.split(',')))
            data_origin.append(data_line)
        frames=[]
        i=0
        while i<len(data_origin):
            data_line=data_origin[i]
            obj=data_line[1]
            frame=data_origin[i:i+obj]
            # frame=np.array(frame)[:,2:].tolist()
            frames.append(frame)
            i+=obj

        for i,frame in enumerate(frames):
            if i==0:
                start=i
                continue
            if (frames[i][0][0]-frames[i-1][0][0]>10) or i==len(frames)-1:
                while start+SEQ_LEN<i:
                    action=frames[start:start+SEQ_LEN]
                    start+=SEQ_LEN
                    for j in range(len(action)):
                        action[j]=np.array(action[j])[:,2:].tolist()
                    data.append(action)
                if i-start>=SEQ_LEN_DROP:
                    action = frames[start:i]
                    for j in range(len(action)):
                        action[j]=np.array(action[j])[:,2:].tolist()
                    data.append(action)
                start=i
        return data

def read_dir(dir,lmdbData):
    print('reading directory: %s' % (dir))
    for file in os.listdir(dir):
        label=file.split('.')[-2].split('_')[-1]
        if label not in LABEL_MAP:
            continue
        path=os.path.join(dir,file)
        data=read_file(path)
        for x in data:
            lmdbData.add(x, LABEL_MAP[label])

def read_data(root=DATA_SOURCE):
    lmdbData=GestureDataset()
    for dir in tqdm(os.listdir(root)):
        path=os.path.join(root, dir)
        read_dir(path,lmdbData)

if __name__=="__main__":
    read_data()

    train_data = GestureDataset(train=True, with_state=True)
    test_data = GestureDataset(train=False, with_state=True)

    print('train dataset length:', train_data.__len__())
    print('test dataset length:', test_data.__len__())
    print(train_data.num_points_in_frame_max, train_data.num_points_in_frame_min)
    print('read dataset testing...')
    for i in range(train_data.__len__()):
        input, label = train_data.__getitem__(i)
        pass
    print('done~!')