import torch
import torch.nn as nn
import os
from mmfall_dataset import Falldataset
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm
from PointGNN_boost import HAR_PointGNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def test_acc(model, dataset, batch_size):
    dataloader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=False)
    model.eval()

    test_correct = 0
    for data in tqdm(dataloader, ascii = True):
        inputs, states, targets = data[0].to(device), data[1].to(device), data[2].to(device)
        # print(inputs.size(),states.size(),targets.size())
        outputs = model(inputs, states)
        _, pred = torch.max(outputs, 1)
        test_correct += torch.sum(pred == targets)
        # print(test_correct)
    del dataloader
    print("Test Accuracy {:.4f}%".format(100.0*test_correct/len(dataset)))



if __name__ == '__main__':
    # set_seed(100)
    batch_size = 10 # 20
    test_batch = 10 # 45
    learning_rate = 0.001

    epoch_num = 2000

    dataset_test = Falldataset(root_falls = "../mmfalldata/DS2_falls", root_normal = "../mmfalldata/DS2_normal" , raw_data = False)

    dataset = Falldataset(root_falls = "../mmfalldata/DS1_4falls", root_normal = "../mmfalldata/DS1_4normal" , raw_data = False)
    train_loader = DataLoader(dataset = dataset,batch_size=batch_size,shuffle=True)

    model = HAR_PointGNN(r = -1, T=3, state_dim=4,frame_num=10, output_dim=2)
    # summary(model,(1,60,250,11))
    model.to(device)

    if os.path.exists('./models/mmFall_PointGNN.pkl'):
        model.load_state_dict(torch.load('./models/mmFall_PointGNN.pkl',map_location = device))
        print("load model sucessfully")

    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(adam, step_size=1, gamma=0.98)

    crossloss = nn.CrossEntropyLoss()
    for epoch in range(1, epoch_num+1):
        # if epoch % 20:
        #     test_acc(model,dataset_test,test_batch)
        model.train()
        epoch_loss = 0
        train_correct = 0

        for batch, data in enumerate(tqdm(train_loader, ascii = True)):
            inputs, states, targets = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(inputs,states)
            _, pred = torch.max(outputs, 1)
            train_correct += torch.sum(pred == targets)
            # print(pred, targets)

            loss = crossloss(outputs,targets)

            adam.zero_grad()
            loss.backward()
            adam.step()
            epoch_loss += loss

            # print('epoch:{}\t batch:{}/{}\t batch loss:{:.4f}'.format(epoch,batch,len(train_loader),loss))
        scheduler.step()
        print('epoch:{}\t epoch loss:{:.4f} \t epoch accuracy:{:.4f} \t learning rate:{}'.format(epoch, epoch_loss, 100.0*train_correct/len(dataset), adam.param_groups[0]['lr']))
        torch.save(model.state_dict(), "./models/mmFall_PointGNN.pkl")

