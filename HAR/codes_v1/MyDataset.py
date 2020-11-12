import lmdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pickle

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.lmdbData=LmdbData(data_dir)

    def __len__(self):
        return self.lmdbData.len()

    def __getitem__(self, idx):
        x,y=self.lmdbData.get(idx)
        #x: (60, 10, 32, 32)
        #y: label 0-4
        if self.transform:
            x=self.transform(x)
        return torch.tensor(x).float(), torch.tensor(y).long()

class LmdbData:
    def __init__(self,dir,map_size = 50):
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

if __name__=="__main__":
    # lmdbData=LmdbData('./lmdbData')
    # print(lmdbData.get(0)[0].shape)
    # print(lmdbData.len())
    train_dataset=MyDataset('./Data/lmdbData_train')
    print(train_dataset.__len__())
    print(train_dataset.__getitem__(0))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True
    )
    # for i, data in enumerate(train_loader):
    #     inputs, labels = data
    #     pass
