import torch
import torch.nn as nn

from Modules.TimeDistributed import TimeDistributed
from Modules.PointNet import PointNetfeat


class Sub_PointNet(nn.Module):
    def __init__(self):
        super(Sub_PointNet, self).__init__()
        self.pointnet = PointNetfeat(global_feat = True)

    def forward(self,x):
        x = x.permute(0, 2, 1)
        out,_,_ = self.pointnet(x)
        return out


class HAR_model(nn.Module):
    def __init__(self, output_dim=5, frame_num = 60):
        super(HAR_model, self).__init__()
        # 1st layer group
        self.pointnet = TimeDistributed(
            nn.Sequential(
                Sub_PointNet()
                )
            )

        # embedding_dim, hidden_size, num_layers
        self.lstm_net = nn.LSTM(1024, 16,num_layers=1, dropout=0,bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(frame_num*2*16,output_dim),
            # nn.Softmax(),
            )

    def forward(self, data):
        data = self.pointnet(data)
        data = data.permute(1,0,2)

         # input(seq_length, batch_size, input_size) [60, N, 1024]
        data,hn = self.lstm_net(data)
        data = data.permute(1,0,2)
        data = data.reshape(data.size(0),-1)
        return self.dense(data)


if __name__ == '__main__':
    a = torch.randn(2,20,40,3)
    model = HAR_model()

    print(model(a))

