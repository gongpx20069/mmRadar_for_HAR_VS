import torch
import torch.nn as nn
import time
from Modules.TimeDistributed import TimeDistributed, PointDistributed

def getdistance(a):
    '''
    input: a.size(N,M,3)
    output: distance.size(N,M,M)
    '''
    b = a.repeat(1,a.size(-2),1)

    c = a.clone()
    c = c.unsqueeze(-2)

    # for i in range(a.size(-2)-1):
    #     d = torch.cat((d,c),dim=-2)
    c = c.repeat(1,1,a.size(-2),1)
    c = c.view(b.size())

    distance = torch.sum((b-c)**2,dim=-1)
    return distance.view(-1,a.size(-2),a.size(-2))

class PointGNN(nn.Module):
    def __init__(self, T=3, r=0.05, conv1d_or_mlp = 'mlp', state_dim = 3):
        super(PointGNN, self).__init__()
        self.T = T
        self.r = r
        self.GetDis = getdistance
        self.conv1d_or_mlp = conv1d_or_mlp
        self.state_dim = state_dim

        self.hardsig = nn.Hardsigmoid()
        
        if self.conv1d_or_mlp == 'mlp':
            # input(N, 42, 3) output(N, 42, 3)
            self.MLP_h = nn.ModuleList([
            PointDistributed(
                nn.Sequential(
                    nn.Linear(state_dim,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128,3),
                    ) 
                ) for i in range(self.T)
            ])
            # input(N, 42, 42, 6) output(N, 42, 42, 128)
            self.MLP_f = nn.ModuleList([
                PointDistributed(
                    nn.Sequential(
                        nn.Linear(state_dim+3,64),
                        nn.ReLU(),
                        nn.Linear(64,128),
                        nn.ReLU(),
                        nn.Linear(128,128),
                        nn.ReLU(),
                        ) 
                    ) for i in range(self.T)
                ])

            # input(N,42,42,128) output(N, 42, 42, )
            self.MLP_r = nn.ModuleList([
                PointDistributed(
                    nn.Sequential(
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                        ) 
                    ) for i in range(self.T)
                ])


            # input(N, 42, 300) output(N, 42, 3)
            self.MLP_g = nn.ModuleList([
                PointDistributed(
                    nn.Sequential(
                        nn.Linear(128,64),
                        nn.ReLU(),
                        nn.Linear(64,32),
                        nn.ReLU(),
                        nn.Linear(32,state_dim),
                        nn.ReLU(),
                        )
                    ) for i in range(self.T)
                ])

        else:
            # input(N, 3, 42) output(N, 3, 42)
            self.MLP_h = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(state_dim,64,1),
                    nn.ReLU(),
                    nn.Conv1d(64,128,1),
                    nn.ReLU(),
                    nn.Conv1d(128,3,1),
                    ) for i in range(self.T)
                ])
            # input(N, 6, 42*42) output(N, 128, 42*42)
            self.MLP_f = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(state_dim + 3, 64, 1),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, 1),
                    nn.ReLU(),
                    nn.Conv1d(128, 128, 1),
                    nn.ReLU()
                    ) for i in range(self.T)
                ])

            # input(N,42,42,128) output(N, 42, 42, )
            self.MLP_r = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(128, 64, 1),
                    nn.ReLU(),
                    nn.Conv1d(64, 32, 1),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, 1),
                    ) for i in range(self.T)
                ])

            # input(N, 128, 42) output(N, 3, 42)
            self.MLP_g = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(128, 64, 1),
                    nn.ReLU(),
                    nn.Conv1d(64, 32, 1),
                    nn.ReLU(),
                    nn.Conv1d(32, state_dim, 1),
                    nn.ReLU()
                    ) for i in range(self.T)
                ])

    def _getTimeEdges(self,x, xi_delta, adj,state):
        xj = x.unsqueeze(-2)
        xj = xj.repeat(1,1, x.size(-2),1)

        xi = xi_delta.unsqueeze(-3)
        # xitemp = xi.clone()

        state = state.unsqueeze(-2)
        # statetemp = state.clone()

        # for i in range(x.size(-2)-1):
        #     xi = torch.cat((xi,xitemp),dim=-2)
        #     state = torch.cat((state,statetemp),dim=-3)
        xi = xi.repeat(1,x.size(-2),1,1)
        state = state.repeat(1, 1,x.size(-2) ,1)

        xi = xj - xi

        xi = torch.cat((xi, state),dim=-1)
        xi = torch.mul(xi, adj)
        return xi

    def _mlp_forward(self,x,state):
        dis = self.GetDis(x)
        # count_edge = torch.where(dis<self.r, torch.full_like(dis, 1), torch.full_like(dis, 0))
        # sum_edge = torch.mean(torch.sum(count_edge,dim=-1))
        # print(count_edge.size(),sum_edge)

        adj = dis < self.r
        adj = adj.unsqueeze(-1)

        for t in range(self.T):
            delta = self.MLP_h[t](state)
            xi_delta = x - delta
            eij_input = self._getTimeEdges(x, xi_delta, adj, state)
            eij_output = self.MLP_f[t](eij_input)
            e_delta = self.MLP_r[t](eij_output).view(-1, eij_output.size(1), eij_output.size(2), 1)

            adj = self.hardsig(e_delta + adj)
            eij_output = torch.mul(eij_output, adj)
            eij_output = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](eij_output) + state
        return state

    def _conv1d_forward(self,x, state):
        dis = self.GetDis(x)
        # count_edge = torch.where(dis<self.r, torch.full_like(dis, 1), torch.full_like(dis, 0))
        # sum_edge = torch.mean(torch.sum(count_edge,dim=-1))
        # print(count_edge.size(),sum_edge)

        adj = dis < self.r
        adj = adj.unsqueeze(-1)

        for t in range(self.T):
            state = state.permute(0,2,1)
            delta = self.MLP_h[t](state)
            delta = delta.permute(0,2,1)
            state = state.permute(0,2,1)

            xi_delta = x - delta
            eij_input = self._getTimeEdges(x, xi_delta, adj, state).permute(0,3,1,2)
            eij_input = eij_input.view(-1,eij_input.size(1),eij_input.size(2)*eij_input.size(3))

            eij_output = self.MLP_f[t](eij_input)
            e_delta = self.MLP_r[t](eij_output)
            eij_output = eij_output.view(-1,eij_output.size(1),x.size(1),x.size(1))

            e_delta = e_delta.view(adj.size())
            adj = self.hardsig(e_delta + adj)

            eij_output = eij_output.permute(0,2,3,1)
            eij_output = torch.mul(eij_output, adj)

            eij_output = torch.max(eij_output,dim=-2)[0]
            eij_output= eij_output.permute(0,2,1)

            state_up = self.MLP_g[t](eij_output)
            state_up = state_up.permute(0,2,1)
            state += state_up

        return state

    def forward(self, x, state):
        if self.conv1d_or_mlp == 'mlp':
            return self._mlp_forward(x, state)
        else:
            return self._conv1d_forward(x, state)


class HAR_PointGNN(nn.Module):
    def __init__(self,r = 0.5,output_dim = 5, T = 3,conv1d_or_mlp = 'mlp',state_dim = 3):
        super(HAR_PointGNN, self).__init__()
        self.pgnn = TimeDistributed(PointGNN(T=T, r=r,conv1d_or_mlp = conv1d_or_mlp, state_dim = state_dim))

        self.lstm_net = nn.LSTM(336, 16,num_layers=1, dropout=0,bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(1920,output_dim),
            nn.Softmax(dim=-1),
            )

    def forward(self,x,state):
        x = self.pgnn(x,state)
        x = x.view(x.size(0),x.size(1),x.size(2)*x.size(3))
        x = x.permute(1,0,2)
        x,hn = self.lstm_net(x)
        x = x.permute(1,0,2)

        x = x.reshape(x.size(0),-1)
        x = self.dense(x)
        return x

def test_point_gnn():
    a = torch.randn(60,42,3).cuda()
    s = torch.randn(60,42,8).cuda()

    model = PointGNN(state_dim = 8).cuda()
    print(model(a,s).size())

def test_conv_mlp():
    a = torch.randn(2,60,42,3).cuda()
    s = torch.randn(2,60,42,8).cuda()
    model_mlp = HAR_PointGNN(r = 0.2,conv1d_or_mlp='mlp',state_dim = 8).cuda().eval()
    model_conv1d = HAR_PointGNN(r = 0.2,conv1d_or_mlp = 'conv1d',state_dim=8).cuda().eval()
    print(model_mlp(a,s).size(),model_conv1d(a,s).size())

    t1 = time.time()
    model_mlp(a,s)
    t2 = time.time()
    model_conv1d(a,s)
    t3 = time.time()

    print('mlp',t2-t1,'conv1d',t3-t2)

if __name__ == '__main__':
    # a = torch.randn(2,60,42,3).cuda()
    # a = torch.randn(60,42,3).cuda()
    # model = PointGNN().cuda()
    # a = torch.randn(2,42,3) # getdistance
    # test_point_gnn()
    test_conv_mlp()
    # a = torch.randn(2,42,3)
    # adj = torch.randn(2,42,42)
    # adj = adj<0.04

    # a = a.unsqueeze(-2)
    # b = a.repeat(1,1,42,1)

    # print(b.size(),adj.size())
    # b[~adj] = b[~adj]+b[~adj]
    # print(b.size())