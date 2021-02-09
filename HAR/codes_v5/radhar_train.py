import torch
import torch.nn as nn
import os
from MyDataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import time
from PointGNN_boost import HAR_PointGNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_acc(model, dataset, batch_size):
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=False)
    model.eval()

    test_correct = 0
    for data in dataloader:
        inputs, states, targets = data[0].to(device), data[1].to(device), data[2].to(device)
        t1 = time.time()
        outputs = model(inputs, states)
        print(time.time()-t1)
        _, pred = torch.max(outputs, 1)
        test_correct += torch.sum(pred == targets)
        # print(test_correct)
    del dataloader
    print("Test Accuracy {:.4f}%".format(100.0*test_correct/len(dataset)))



if __name__ == '__main__':
    batch_size = 5
    test_batch = 1
    learning_rate = 0.0001

    epoch_num = 200

    dataset_test = MyDataset('../Data/lmdbData_test', padding='zero')

    dataset = MyDataset('../Data/lmdbData_train',padding='zero')
    train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True)

    model = HAR_PointGNN(r = 5, T=3, state_dim=8,frame_num=60)
    model.to(device)

    if os.path.exists('./models/HAR_PointGNN.pkl'):
        model.load_state_dict(torch.load('./models/HAR_PointGNN.pkl',map_location = device))
        print("load model sucessfully")

    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1, gamma=0.8)

    crossloss = nn.CrossEntropyLoss()
    for epoch in range(98, epoch_num+1):
        test_acc(model,dataset_test,test_batch)
        model.train()
        epoch_loss = 0

        for batch, data in enumerate(train_loader):
            inputs, states, targets = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(inputs,states)
            loss = crossloss(outputs,targets)

            adam.zero_grad()
            loss.backward()
            adam.step()
            epoch_loss += loss

            # print('epoch:{}\t batch:{}/{}\t batch loss:{:.4f}'.format(epoch,batch,len(train_loader),loss))
        scheduler.step()
        print('epoch:{}\t epoch loss:{:.4f} \t learning rate:{}'.format(epoch, epoch_loss, adam.param_groups[0]['lr']))
        torch.save(model.state_dict(), "./models/HAR_PointGNN.pkl")

