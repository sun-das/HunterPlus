import torch
import torch.nn as nn
from .constants import *
from .npn import *
from .gnn import *

import dgl
from dgl.nn.pytorch import GraphConv
from sys import argv

class energy_50(nn.Module):
    def __init__(self):
        super(energy_50, self).__init__()
        self.name = "energy_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 51, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        return x

class energy_latency_50(nn.Module):
    def __init__(self):
        super(energy_latency_50, self).__init__()
        self.name = "energy_latency_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 52, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency_10(nn.Module):
    def __init__(self):
        super(energy_latency_10, self).__init__()
        self.name = "energy_latency_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 12, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency2_10(nn.Module):
    def __init__(self):
        super(energy_latency2_10, self).__init__()
        self.name = "energy_latency2_10"
        self.find = nn.Sequential(
            nn.Linear(10 * 14, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latency2_50(nn.Module):
    def __init__(self):
        super(energy_latency2_50, self).__init__()
        self.name = "energy_latency2_50"
        self.find = nn.Sequential(
            nn.Linear(50 * 54, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 64), 
            nn.Tanhshrink(),
            nn.Linear(64, 2),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latencyGNN_50(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_50, self).__init__()
        self.name = "energy_latencyGNN_50"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(50 * 2 * self.emb + 50 * 52, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; graph.ndata['h'] = data
        for layer in self.grapher:
            x = layer(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

#Original - remove _1 to get it working properly
class energy_latencyGNN_10_1(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_10, self).__init__()
        self.name = "energy_latencyGNN_10"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(10 * 2 * self.emb + 10 * 12, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; graph.ndata['h'] = data
        for layer in self.grapher:
            x = layer(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

# This is for BiGRU with 4 hosts
class energy_latencyGNN_4_1(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4 * 2 * 2 * self.emb + 4 * 6 , 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f, layer_b in zip(self.grapher, self.grapher_b):
            x = layer_f(graph, x)
            xx = layer_b(graph, xx)
        x = x.view(-1)
        xx = xx.view(-1)
        x = torch.cat((x, xx))
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

class energy_latencyGNN_1_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_1_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_1_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * self.emb + 4 * 6 , 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        #self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            #self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f in self.grapher:
            x = layer_f(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x


#The model below is the full proper BiGRU implementation. It is useful for 10 hosts.
class energy_latencyGNN_10(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_10, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_10"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(10 * 2 * 2 * self.emb + 10 * 12 , 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f, layer_b in zip(self.grapher, self.grapher_b):
            x = layer_f(graph, x)
            xx = layer_b(graph, xx)
        x = x.view(-1)
        xx = xx.view(-1)
        x = torch.cat((x, xx))
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
#################################################
# The class below is for testing the CNN model

class energy_latencyCNN_4(nn.Module):
    def __init__(self):
        super(energy_latencyCNN_4, self).__init__()
        self.name = "energy_latencyCNN_4"
        self.find = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = (6, 6), padding = 3),
            nn.ReLU(),

            nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = (5, 5), padding = 3),
            nn.ReLU(),

            nn.Conv2d(in_channels = 20, out_channels = 30, kernel_size = (4, 4), padding = 2),
            nn.ReLU(),

            nn.Conv2d(in_channels = 30, out_channels = 40, kernel_size = (3, 3), padding = 2),
            nn.ReLU(),

            nn.Conv2d(in_channels = 40, out_channels = 50, kernel_size = (2, 2), padding = 1),
            nn.ReLU(),

            #FOR INTENSE LOAD
            nn.Conv2d(in_channels = 50, out_channels = 40, kernel_size = (2, 2), padding = 1),
            nn.ReLU(),

            nn.Conv2d(in_channels = 40, out_channels = 30, kernel_size = (3, 3), padding = 2),
            nn.ReLU(),

            nn.Conv2d(in_channels = 30, out_channels = 20, kernel_size = (4, 4), padding = 3),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(in_features = 17160, out_features = 2),

            nn.Sigmoid()
        )

    def forward(self, x):
        #print(x)
        x = self.find(x)
        #print(len(x))
        #print(x[0])
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0][0] + Coeff_Latency*x[0][1]
        return x

#################################################
# Classes below are for testing 3 Linear Layers

#Single GRU, 128 inputs, 3 GRU
class energy_latencyGNN_GRU128L3_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_GRU128L3_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_GRU128L3_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * self.emb + 4 * 5 * 10, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        #self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            #self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f in self.grapher:
            x = layer_f(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

# Single GRU, 64 input, 3 GRU layers, 
class energy_latencyGNN_GRU64L3_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_GRU64L3_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_GRU64L3_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * self.emb + 4 * 5 * 10 , 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        #self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            #self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f in self.grapher:
            x = layer_f(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

# Single GRU, 32 input, 3 GRU layers
class energy_latencyGNN_GRU32L3_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_GRU32L3_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_GRU32L3_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * self.emb + 4 * 5 * 10 , 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16), 
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        #self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            #self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f in self.grapher:
            x = layer_f(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x



#Bi GRU, 128 inputs, 3 GRU
class energy_latencyGNN_BiGRU128L3_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_BiGRU128L3_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_BiGRU128L3_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * 2 * self.emb + 4 * 7 * 10 , 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):

        #print("graph.ndata['h']:\n")
        #print(graph.ndata['h'])
        #print("\ndata:\n")
        #print(data)
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f, layer_b in zip(self.grapher, self.grapher_b):
            x = layer_f(graph, x)
            xx = layer_b(graph, xx)
        x = x.view(-1)
        xx = xx.view(-1)
        x = torch.cat((x, xx))
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

# BiGRU, 64 input, 3 GRU layers, 
class energy_latencyGNN_BiGRU64L3_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_BiGRU64L3_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_BiGRU64L3_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * 2 * self.emb + 4 * 7 * 10 , 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f, layer_b in zip(self.grapher, self.grapher_b):
            x = layer_f(graph, x)
            xx = layer_b(graph, xx)
        x = x.view(-1)
        xx = xx.view(-1)
        x = torch.cat((x, xx))
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

# Single GRU, 32 input, 3 GRU layers
class energy_latencyGNN_BiGRU32L3_4(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_BiGRU32L3_4, self).__init__()
        print("I am the BiGRU at your service!")
        self.name = "energy_latencyGNN_BiGRU32L3_4"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(4  * 2 * 2 * self.emb + 4 * 7 * 10 , 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16), 
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher_b = nn.ModuleList()

        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        self.grapher_b.append(BackwardGatedRGCNLayer(1, self.emb, activation = nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))
            self.grapher_b.append(BackwardGatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; xx = data; graph.ndata['h'] = data
        for layer_f, layer_b in zip(self.grapher, self.grapher_b):
            x = layer_f(graph, x)
            xx = layer_b(graph, xx)
        x = x.view(-1)
        xx = xx.view(-1)
        x = torch.cat((x, xx))
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x




#################################################
class stochastic_energy_latency_50(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency_50, self).__init__()
        self.name = "stochastic_energy_latency_50"
        self.find = nn.Sequential(
            NPNLinear(50 * 52, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency2_50(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency2_50, self).__init__()
        self.name = "stochastic_energy_latency2_50"
        self.find = nn.Sequential(
            NPNLinear(50 * 54, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency_10(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency_10, self).__init__()
        self.name = "stochastic_energy_latency_10"
        self.find = nn.Sequential(
            NPNLinear(10 * 12, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s

class stochastic_energy_latency2_10(nn.Module):
    def __init__(self):
        super(stochastic_energy_latency2_10, self).__init__()
        self.name = "stochastic_energy_latency2_10"
        self.find = nn.Sequential(
            NPNLinear(10 * 14, 128, False),
            NPNRelu(),
            NPNLinear(128, 128),
            NPNRelu(),
            NPNLinear(128, 64), 
            NPNRelu(),
            NPNLinear(64, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.find(x)
        if not('train' in argv[0] and 'train' in argv[2]):
            return x + UCB_K * s
        return x, s
