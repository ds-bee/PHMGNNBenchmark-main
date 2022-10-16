import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool  # noqa


class GCN(torch.nn.Module):
    def __init__(self, feature, out_channel):
        super(GCN, self).__init__()

        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        #layers
        self.GConv1 = GCNConv(feature,1024)
        self.bn1 = BatchNorm(1024)

        self.GConv2 = GCNConv(1024,1024)
        self.bn2 = BatchNorm(1024)

        # self.fc = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        # self.dropout = nn.Dropout(0.2)
        self.conv3 = GCNConv(1024, 512)
        self.bn3 = BatchNorm(512)

        self.fc1 = nn.Sequential(nn.Linear(512, out_channel))

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x.requires_grad = True
        self.input = x
        print(x.shape)
        x = self.GConv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        h1 = x
        print(h1.shape)
        x = self.GConv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        h2 = x

        with torch.enable_grad():
            self.final_conv_acts = self.conv3(h2, edge_index, edge_weight)

        self.final_conv_acts.register_hook(self.activations_hook)

        x = F.relu(self.final_conv_acts)  # h3

        h4 = global_mean_pool(x, data.batch)

        # x = self.fc(x)
        # x = self.dropout(x)

        x = self.fc1(h4)

        return x
if __name__ == '__main__':
    model = GCN(1024,10)
    print(model)