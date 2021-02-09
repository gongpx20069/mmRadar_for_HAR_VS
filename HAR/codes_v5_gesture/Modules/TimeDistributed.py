import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def __multi_time(self,size):
        size_temp = list(size)
        size_temp = [size_temp[0]*size_temp[1]]+size_temp[2:]
        return tuple(size_temp)

    def __dist_time(self,size,batch,time_dim):
        size_temp = list(size)
        size_temp = [batch,time_dim]+size_temp[1:]
        return tuple(size_temp)

    def forward(self, *args):
        # Squash samples and timesteps into a single axis
        x_reshape = (x.contiguous().view(self.__multi_time(x.size())) for x in args)  # (samples * timesteps, input_size)

        y = self.module(*x_reshape)

        y = y.contiguous().view(self.__dist_time(y.size(),args[0].size(0),args[0].size(1)))  # (samples, timesteps, output_size)

        return y

class PointDistributed(nn.Module):
    def __init__(self, module):
        super(PointDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        original = list(x.size())
        x = x.view(-1, original[-1])
        y = self.module(x)

        original[-1] = -1
        y = y.view(original)
        return y


if __name__ == '__main__':
    a = torch.randn(1,60,10,32,32)
    model = TD_CNN_LSTM()

    print(model(a))

