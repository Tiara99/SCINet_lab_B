import torch as th
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class NN_block(nn.Module):
    def __init__(self, channel=1, hidden=5, kernel=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReplicationPad1d(kernel // 2 * 2),
            nn.Conv1d(channel, channel * hidden, kernel),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Conv1d(channel * hidden, channel, kernel),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.block(x)


class SCI_block(nn.Module):
    def __init__(self):
        super().__init__()

        self.psi = NN_block()
        self.theta = NN_block()
        self.mi = NN_block()
        self.ro = NN_block()

    def forward(self, f):
        f_odd = f[:, :, 1::2]
        f_even = f[:, :, 0::2]

        F_even_s = f_even * th.exp(self.psi(f_odd))
        F_odd_s = f_odd * th.exp(self.theta(f_even))

        F_even = F_even_s - self.mi(F_odd_s)
        F_odd = F_odd_s + self.ro(F_even_s)

        return F_even, F_odd


class SCINet_block(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 layers
        self.block_1 = SCI_block()

        self.block_2_0 = SCI_block()
        self.block_2_1 = SCI_block()

        self.block_3_0 = SCI_block()
        self.block_3_1 = SCI_block()
        self.block_3_2 = SCI_block()
        self.block_3_3 = SCI_block()

        self.fc = nn.Linear(40, 10)
        self.fc1 = nn.Linear(40, 40)

    def forward(self, x):
        out_2_0, out_2_1 = self.block_1(x)
        out_2_0_0, out_2_0_1 = self.block_2_0(out_2_0)
        out_2_1_0, out_2_1_1 = self.block_2_1(out_2_1)

        out_3_0_0, out_3_0_1 = self.block_3_0(out_2_0_0)
        out_3_1_0, out_3_1_1 = self.block_3_1(out_2_0_1)
        out_3_2_0, out_3_2_1 = self.block_3_2(out_2_1_0)
        out_3_3_0, out_3_3_1 = self.block_3_3(out_2_1_1)

        # conbine back
        out_stacked = th.cat(
            [
                out_3_0_0,
                out_3_0_1,
                out_3_1_0,
                out_3_1_1,
                out_3_2_0,
                out_3_2_1,
                out_3_3_0,
                out_3_3_1,
            ],
            dim=1,
        )
        out = out_stacked.transpose(dim0=1, dim1=2).reshape(x.shape[0], 1, -1)

        out_1 = self.fc1(out)
        add_x = out_1 + x
        out_final = self.fc(add_x)

        return out_final


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(40, 10)

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        return out


# ## Preprocess data ### #
class MinMax:
    def __init__(self, data):
        self.max = data.max()
        self.min = data.min()

        scaled = (data - self.min) / (self.max - self.min)
        self.scaled = scaled

    def transform(self):
        return self.scaled

    def inverse(self, data):
        standardize = data * (self.max - self.min) + self.min
        return standardize


FILENAME = "MSFT.csv"

data = pd.read_csv(FILENAME)
open_data = data["Open"].to_numpy()
min_max = MinMax(open_data[:40_000])


def load_data():
    open_data_scaled = min_max.transform()

    # prepair train sequence
    window_sz = 40
    future_sz = 10

    net_data = []
    for i in range(window_sz, len(open_data_scaled) - future_sz - 1):
        roi_x = open_data_scaled[i - window_sz : i]
        roi_y = open_data_scaled[i : i + future_sz]

        net_data.append(
            [
                roi_x.reshape(1, -1).astype(np.float32),
                roi_y.reshape(1, -1).astype(np.float32),
            ]
        )

    return net_data


def create_loaders(data):
    split = int(0.8 * len(data))
    return DataLoader(data[:split], batch_size=64, shuffle=True), DataLoader(
        data[:split], batch_size=256
    )


def evaluate(net, test_loader):
    def loss(a, b):
        # a_raw = min_max.inverse(a)
        # b_raw = min_max.inverse(b)
        return (a - b).abs().sum()

    net.eval()
    with th.no_grad():
        total_loss = 0
        for x, y in test_loader:
            y_est = net(x)
            total_loss += loss(y_est, y)

    norm_loss = total_loss / len(test_loader)
    print(f"Total loss: {norm_loss : .4f}")


def train(net, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = th.optim.AdamW(net.parameters())
    net.train()

    epochs = 50
    for eps in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()

            y_est = net(x)
            loss = criterion(y_est, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        norm_loss = total_loss / len(train_loader)

        end = "\t" if (eps + 1) % 10 == 0 else "\n"
        print(f"[{eps+1}/{epochs}] Total loss: {norm_loss : .4e}", end=end)
        if (eps + 1) % 10 == 0:
            evaluate(net, test_loader)
