import torch
import torch.nn as nn
import os
from MyDataset import MyDataset
from torch.utils.data import Dataset, DataLoader

from TD_CNN_LSTM import TD_CNN_LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def test_acc(model, dataset):
    dataloader = DataLoader(dataset = dataset,batch_size=4,shuffle=False)
    model.eval()

    test_correct = 0
    for data in dataloader:
        input, target = data[0].to(device), data[1].to(device)
        output = model(input)

        _, pred = torch.max(output, 1)
        test_correct += torch.sum(pred == target)
    del dataloader
    print("Test Accuracy {:.4f}%".format(100.0*test_correct/len(dataset)))



if __name__ == '__main__':
    batch_size = 7
    learning_rate = 0.00001

    epoch_num = 100

    dataset_test = MyDataset('./Data/lmdbData_test')

    dataset = MyDataset('./Data/lmdbData_train')
    train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True)

    model = TD_CNN_LSTM()
    model.to(device)

    if os.path.exists('./models/TD_CNN_LSTM.pkl'):
        model.load_state_dict(torch.load('./models/TD_CNN_LSTM.pkl',map_location = device))
        print("load generator model sucessfully")

    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1, gamma=0.98)

    crossloss = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        test_acc(model,dataset_test)
        model.train()
        scheduler.step()
        epoch_loss = 0

        for batch, data in enumerate(train_loader):
            input, target = data[0].to(device), data[1].to(device)
            output = model(input)
            loss = crossloss(output,target)

            adam.zero_grad()
            loss.backward()
            adam.step()
            epoch_loss += loss

            print('epoch:{}\t batch:{}/{}\t batch gen loss:{:.4f}'.format(epoch,batch,len(train_loader),loss))
        print('epoch:{}\t epoch gen loss:{:.4f}'.format(epoch,epoch_loss))
        torch.save(model.state_dict(), "./models/TD_CNN_LSTM.pkl")

