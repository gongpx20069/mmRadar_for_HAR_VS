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
    def __init__(self, T=3, r=0.05):
        super(PointGNN, self).__init__()
        self.T = T
        self.r = r
        self.GetDis = getdistance
        # input(N, 42, 3) output(N, 42, 3)
        self.MLP_h = nn.ModuleList([
            PointDistributed(
                nn.Sequential(
                    nn.Linear(3,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128,3),
                    ) 
                ) for i in range(self.T)
            ])
        # input(N, 42, 42, 6) output(N, 42, 42, 300)
        self.MLP_f = nn.ModuleList([
            PointDistributed(
                nn.Sequential(
                    nn.Linear(6,64),
                    nn.ReLU(),
                    nn.Linear(64,128),
                    nn.ReLU(),
                    nn.Linear(128,128),
                    nn.ReLU(),
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
                    nn.Linear(32,3),
                    nn.ReLU(),
                    )
                ) for i in range(self.T)
            ])

    def getTimeEdges(self,x,adj,state):
        xj = x.unsqueeze(-2)
        xj = xj.repeat(1,1,x.size(-2),1)

        xi = x.unsqueeze(-2)
        xitemp = xi.clone()

        state = state.unsqueeze(-3)
        statetemp = state.clone()

        # for i in range(x.size(-2)-1):
        #     xi = torch.cat((xi,xitemp),dim=-2)
        #     state = torch.cat((state,statetemp),dim=-3)
        xi = xi.repeat(1,1, x.size(-2),1)
        state = state.repeat(1, x.size(-2) ,1 ,1)

        xi[~adj] = 0
        xi[adj] = xj[adj] - xi[adj]

        state[~adj] = 0
        xi = torch.cat((xi, state),dim=-1)
        return xi


    def forward(self, x):
        dis = self.GetDis(x)
        # count_edge = torch.where(dis<self.r, torch.full_like(dis, 1), torch.full_like(dis, 0))
        # sum_edge = torch.mean(torch.sum(count_edge,dim=-1))
        # print(count_edge.size(),sum_edge)

        adj = dis < self.r

        state = x.clone()

        for t in range(self.T):
            delta = self.MLP_h[t](state)
            xi_delta = x - delta
            eij_input = self.getTimeEdges(xi_delta, adj, state)
            eij_output = self.MLP_f[t](eij_input)
            eij_output[~adj] = 0
            eij_output = torch.max(eij_output,dim=-2)[0]

            state = self.MLP_g[t](eij_output) + state

        return state

class HAR_PointGNN(nn.Module):
    def __init__(self,r = 0.5,output_dim = 5, T = 3):
        super(HAR_PointGNN, self).__init__()
        self.pgnn = TimeDistributed(PointGNN(T=T, r=r))

        self.lstm_net = nn.LSTM(126, 16,num_layers=1, dropout=0,bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(1920,output_dim),
            nn.Softmax(dim=-1),
            )

    def forward(self,x):
        x = self.pgnn(x)
        x = x.view(x.size(0),x.size(1),x.size(2)*x.size(3))

        x = x.permute(1,0,2)
        x,hn = self.lstm_net(x)
        x = x.permute(1,0,2)

        x = x.reshape(x.size(0),-1)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    a = torch.randn(3,60,42,3).cuda()
    # a = torch.randn(60,42,3).cuda()
    # model = PointGNN().cuda()
    # a = torch.randn(2,42,3) # getdistance

    model = HAR_PointGNN(r = 0.2).cuda()
    t1 = time.time()
    print(model(a).size())

    t2 = time.time()
    print(model(a).size())

    t3 = time.time()

    print(t3-t2, t2-t1)

    # a = torch.randn(2,42,3)
    # adj = torch.randn(2,42,42)
    # adj = adj<0.04

    # a = a.unsqueeze(-2)
    # b = a.repeat(1,1,42,1)

    # print(b.size(),adj.size())
    # b[~adj] = b[~adj]+b[~adj]
    # print(b.size())