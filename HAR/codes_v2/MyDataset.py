import random

import lmdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pickle

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, padding='repeat', extra=True):
        # padding: {repeat, zero}
        self.padding=padding
        self.extra=extra # feature including range, velocity... if set True
        self.data_dir = data_dir
        self.transform = transform
        self.lmdbData=LmdbData(data_dir)
        self.paddingto=42
        self.padding_value=0

    def __len__(self):
        return self.lmdbData.len()

    def __getitem__(self, idx):
        x,y=self.lmdbData.get(idx)
        # x: (60, k, 3)  or (60,k,8)
        # y: label 0-4
        data=np.zeros((len(x),self.paddingto,8))
        for i, frame in enumerate(x):
            if self.padding=='zero':
                frame.extend([[self.padding_value] * 8 for i in range(self.paddingto - len(frame))])
                data[i]=np.array(frame)
            elif self.padding=='repeat':
                ids=list(range(len(frame)))
                pad_ids=[random.choice(ids) for j in range(self.paddingto - len(frame))]
                data[i,:len(frame)]=np.array(frame)
                data[i,len(frame):]=data[i,pad_ids]
            else:
                print('wrang padding type!')
                exit(0)
        if not self.extra:
            data=data[:,:,:3]
        if self.transform:
            data=self.transform(data)
        else:
            data=torch.tensor(data).float()
        label=torch.tensor(y).long()
        return data, label

class LmdbData:
    def __init__(self,dir,map_size = 1):
        self.env = lmdb.open(dir, map_size=map_size*1024*1024*1024) #100G
        self.next=0
        self.action_label={
            "boxing":0,
            "jack":1,
            "jump":2,
            "squats":3,
            "walk":4
        }
        pass
    def len(self):
        txn = self.env.begin()
        len = int(txn.get('len'.encode()).decode())
        return len

    def get(self, id):
        txn = self.env.begin()
        x = pickle.loads(txn.get(('x_%d'%(id)).encode()))
        y = int(txn.get(('y_%d' % (id)).encode()).decode())
        return x,y

    def add(self,x,y):
        txn = self.env.begin(write=True)
        txn.put(('x_%d'%(self.next)).encode(), pickle.dumps(x))
        txn.put(('y_%d' % (self.next)).encode(), str(self.action_label[y]).encode())
        txn.put('len'.encode(), str(self.next+1).encode())
        txn.commit()
        self.next+=1

    def put(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k, v)
        txn.commit()

if __name__=="__main__":
    # lmdbData=LmdbData('./lmdbData')
    # print(lmdbData.get(0)[0].shape)
    # print(lmdbData.len())
    train_dataset=MyDataset('./Data/lmdbData_train')
    print(train_dataset.__len__())
    # print(train_dataset.__getitem__(0))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True
    )
    for i, data in enumerate(train_loader):
        inputs, labels = data

        print(inputs.size(), labels.size())
        break
