import torch
import torch.nn as nn
import os
from gesture_dataset import GestureDataset
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm
from PointGNN_boost import HAR_PointGNN

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def test_acc(model, dataset, batch_size):
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=False)
    model.eval()
    test_correct = 0
    for data in tqdm(dataloader):
        inputs, states, targets = data[0].cuda(), data[1].cuda(), data[2].cuda()
        # states = torch.cat((states, states), -1)
        a,b=torch.isnan(inputs).sum(),torch.isnan(states).sum()
        if a>0 or b>0:
            stop
        # print(inputs.size(),states.size(),targets.size())
        outputs = model(inputs, states)

        _, pred = torch.max(outputs, 1)
        test_correct += torch.sum(pred == targets)
        # print(test_correct)
    del dataloader
    print("Test Accuracy {:.4f}%".format(100.0*test_correct/len(dataset)))


if __name__ == '__main__':
    batch_size =  30 #40 6
    test_batch =  50 # 80 12
    learning_rate = 0.005

    epoch_num = 200
    dataset_test = GestureDataset(train=False, with_state=True)

    dataset = GestureDataset(train=True, with_state=True)
    train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True)

    model = HAR_PointGNN(r = -1, T=3, state_dim=5,frame_num=20, output_dim=4, extend=128, lstm_hidden = 16)
    # summary(model,(1,60,250,11))
    model = nn.DataParallel(model)

    if os.path.exists('./models/gesture_PointGNN.pkl'):
        model.load_state_dict(torch.load('./models/gesture_PointGNN.pkl'))
        print("load model sucessfully")

    model.cuda()

    adam = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=2, gamma=0.7)

    crossloss = nn.CrossEntropyLoss()
    for epoch in range(1, epoch_num+1):
        test_acc(model,dataset_test,test_batch)
        model.train()
        epoch_loss = 0
        train_correct = 0

        for batch, data in enumerate(tqdm(train_loader)):
            inputs, states, targets = data[0].cuda(), data[1].cuda(), data[2].cuda()
            # states = torch.cat((states, states), -1)
            a,b=torch.isnan(inputs).sum(),torch.isnan(states).sum()
            if a>0 or b>0:
                stop
            outputs = model(inputs,states)

            _, pred = torch.max(outputs, 1)
            train_correct+=torch.sum(pred==targets)

            loss = crossloss(outputs,targets)
            adam.zero_grad()
            loss.backward()
            adam.step()

            epoch_loss += loss.item()

            # print('epoch:{}\t batch:{}/{}\t batch loss:{:.4f}'.format(epoch,batch,len(train_loader),loss))
        scheduler.step()
        print('epoch:{}\t epoch loss:{:.4f} \t train Accuracy:{:.4f}% \t learning rate:{}'.format(epoch, epoch_loss, 100.0*train_correct/len(dataset),adam.param_groups[0]['lr']))
        torch.save(model.state_dict(), "./models/gesture_PointGNN.pkl")

