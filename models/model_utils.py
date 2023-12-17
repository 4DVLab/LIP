import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        #print(x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
    
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        #print(x.shape)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class GruModule(nn.Module):
    """
    Definition of specific GRU module.
    """
    def __init__(
        self,
        n_layers=2, input_size=172, hidden_size=512, output_size=512,
        bidirectional=True, use_residual=False, feed_forward=True,
        dropout=0.5
    ):
        """
        Auto-called initializer of the specific GRU module.
        """
        super(GruModule, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )

        self.linear = None
        if bidirectional:
            hidden_size = hidden_size*2
        if use_residual:
            self.linear = nn.Linear(hidden_size, input_size)
        elif feed_forward:
            self.linear = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=dropout)
        self.use_residual = use_residual

    def forward(self, input, init_state=None):
        """
        Forward propagation of this specific GRU module
        """
        N, L, D = input.shape
        hidden_result, _ = self.gru(input.float(), init_state)
        output = None
        if self.linear:
            output = F.elu(hidden_result)
            output = self.linear(output.contiguous().view(-1, output.shape[-1]))
            output = output.view(N, L, output.shape[-1])
        if self.use_residual and output.shape[-1] == input.shape[-1]:
            output += input
        
        return output

class Tracker(nn.Module):
    """
    Definition of Tracker module.
    """
    def __init__(
        self,
        input_size=172,
        hidden_size=512,
        output_size=512,
        dropout=0.5
    ):
        """
        Auto-called initializer of the Tracker module.
        """
        super(Tracker, self).__init__()
        self.encoder1 = GruModule(
            n_layers=1,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            bidirectional=False,
            use_residual=False,
            feed_forward=True,
        )

        self.encoder2 = GruModule(
            n_layers=1,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            bidirectional=False,
            use_residual=True,
            feed_forward=True,
        )

        self.encoder3 = GruModule(
            n_layers=1,
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=output_size,
            bidirectional=False,
            use_residual=False,
            feed_forward=True,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, init_state=None):
        N, L, D = input.shape
        encoded_feature1 = self.dropout(self.encoder1(input, init_state))
        encoded_feature2 = self.dropout(self.encoder2(encoded_feature1, init_state))
        output = self.encoder3(encoded_feature2, init_state)

        return output
    
        