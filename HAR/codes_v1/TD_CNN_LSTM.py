import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def __multi_time(self,size):
        size_temp = list(size)
        size_temp = [size_temp[0]*size_temp[1]]+size_temp[2:]
        return tuple(size_temp)

    def __dist_time(self,size,batch,time_dim):
        size_temp = list(size)
        size_temp = [batch,time_dim]+size_temp[1:]
        return tuple(size_temp)

    def forward(self, x):
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(self.__multi_time(x.size()))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        y = y.contiguous().view(self.__dist_time(y.size(),x.size(0),x.size(1)))  # (samples, timesteps, output_size)

        return y


class TD_CNN_LSTM(nn.Module):
    def __init__(self, output_dim=5):
        super(TD_CNN_LSTM, self).__init__()
        # 1st layer group
        self.conv3d_1 = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(1,32,kernel_size=(3,3,3),stride=(1,1,1),padding = 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                )
            )

        # 2nd layer group
        self.conv3d_2 = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(1,1,1),padding = 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                )
            )

        self.maxpool = TimeDistributed(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        # 3rd layer group
        self.conv3d_3_1 = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(1,1,1),padding = 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                )
            )
        self.conv3d_3_2 = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(1,1,1),padding = 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                )
            )
        self.conv3d_3_3 = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(1,1,1),padding = 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                )
            )
        self.conv3d_3_4 = TimeDistributed(
            nn.Sequential(
                nn.Conv3d(32,32,kernel_size=(3,3,3),stride=(1,1,1),padding = 1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                )
            )
        self.lstm_net = nn.LSTM(512, 16,num_layers=1, dropout=0,bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(32*60,output_dim),
            nn.Softmax(),
            )

    def forward(self, data):
        data = data.unsqueeze(2)
        data = self.conv3d_1(data)
        data = self.conv3d_2(data)
        data = self.maxpool(data)

        data = self.conv3d_3_1(data)
        data = self.conv3d_3_2(data)
        data = self.maxpool(data)

        data = self.conv3d_3_3(data)
        data = self.conv3d_3_4(data)
        # print(data.size())
        data = self.maxpool(data)

        data = data.permute(1,0,2,3,4,5)
        data = data.view(data.size(0),data.size(1),-1)

        data,hn = self.lstm_net(data)
        data = data.permute(1,0,2)
        data = data.reshape(data.size(0),-1)

        return self.dense(data)


if __name__ == '__main__':
    a = torch.randn(1,60,10,32,32)
    model = TD_CNN_LSTM()

    print(model(a))

