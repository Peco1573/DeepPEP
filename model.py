import torch.nn as nn
import torch.nn.functional as F
import torch


class ESM2_FC(nn.Module):
    def __init__(self):
        super(ESM2_FC, self).__init__()
        self.linear1 = nn.Linear(5120, 2)

    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        return x


class ESM2_MLP(nn.Module):
    def __init__(self):
        super(ESM2_MLP, self).__init__()
        self.linear_1 = nn.Linear(5120, 2048)
        self.linear_2 = nn.Linear(2048, 1024)
        self.linear_3 = nn.Linear(1024, 512)
        self.linear_4 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        x = F.sigmoid(self.linear_4(x))
        return x


class ESM2_LSTM(nn.Module):
    def __init__(self):
        super(ESM2_LSTM, self).__init__()
        self.biLSTM = nn.LSTM(input_size=5120, hidden_size=256, num_layers=2, bidirectional=False, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(256, 2)

    def forward(self, x):
        x, _ = self.biLSTM(x.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = torch.sigmoid(self.linear(x[:, -1, :]))
        return x


class ESM2_biLSTM(nn.Module):
    def __init__(self):
        super(ESM2_biLSTM, self).__init__()
        self.biLSTM = nn.LSTM(input_size=5120, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(512, 2)

    def forward(self, x):
        x, _ = self.biLSTM(x.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = torch.sigmoid(self.linear(x[:, -1, :]))
        return x


class ESM2_biLSTM_MLP(nn.Module):
    def __init__(self):
        super(ESM2_biLSTM_MLP, self).__init__()
        self.biLSTM = nn.LSTM(input_size=5120, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 2)

    def forward(self, x):
        x, _ = self.biLSTM(x.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.linear1(x[:, -1, :]))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x
