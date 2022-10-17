import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,  BatchNorm # noqa
from torch_geometric.nn import TopKPooling,  EdgePooling, ASAPooling, SAGPooling, global_mean_pool

class GCN1(torch.nn.Module):
    def __init__(self, feature, out_channel,pooltype):
        super(GCN1, self).__init__()

        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        self.pool1, self.pool2 = self.poollayer(pooltype)

        self.GConv1 = GCNConv(feature,1024)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = GCNConv(1024,1024)
        self.bn2 = BatchNorm(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data, pooltype):
        h0, edge_index, batch= data.x, data.edge_index, data.batch
        h0.requires_grad = True
        self.input = h0
        x = self.GConv1(h0, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x, edge_index, batch = self.poolresult(self.pool1,pooltype,x, edge_index, batch)

        x1 = global_mean_pool(x, batch)

        x = self.GConv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x, edge_index, batch = self.poolresult(self.pool2, pooltype, x, edge_index, batch)
        x2 = global_mean_pool(x, batch)

        x = x1 + x2
        x = self.fc(x)
        x = self.dropout(x)
        x = self.fc1(x)

        return x

    def poollayer(self, pooltype):

        self.pooltype = pooltype

        if self.pooltype == 'TopKPool':
            self.pool1 = TopKPooling(1024)
            self.pool2 = TopKPooling(1024)
        elif self.pooltype == 'EdgePool':
            self.pool1 = EdgePooling(1024)
            self.pool2 = EdgePooling(1024)
        elif self.pooltype == 'ASAPool':
            self.pool1 = ASAPooling(1024)
            self.pool2 = ASAPooling(1024)
        elif self.pooltype == 'SAGPool':
            self.pool1 = SAGPooling(1024)
            self.pool2 = SAGPooling(1024)
        else:
            print('Such graph pool method is not implemented!!')

        return self.pool1, self.pool2

    def poolresult(self,pool,pooltype,x,edge_index,batch):

        self.pool = pool

        if pooltype == 'TopKPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'EdgePool':
            x, edge_index, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'ASAPool':
            x, edge_index, _, batch, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        elif pooltype == 'SAGPool':
            x, edge_index, _, batch, _, _ = self.pool(x=x, edge_index=edge_index, batch=batch)
        else:
            print('Such graph pool method is not implemented!!')

        return x, edge_index, batch