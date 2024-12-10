import numpy as np
import math
import needle.nn as nn
import needle as ndl
import sys
sys.path.append('./python')
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        self.conv2d = nn.Conv(in_channels, out_channels,
                              kernel_size, stride, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(dim=out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.cb1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.cb2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)

        self.cb3 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.cb4 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)

        self.cb5 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.cb6 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)

        self.cb7 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.cb8 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)

        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x):
        # print(f'x shape: {x.shape}')  # (2, 3, 32, 32)
        # BEGIN YOUR SOLUTION
        x1 = self.cb1(x)  # (2, 16, 8, 8)
        x2 = self.cb2(x1)  # (2, 32, 4, 4)
        #print(f'x2 shape: {x2.shape}')

        x3 = self.cb3(x2)  # (2, 32, 4, 4)
        x4 = self.cb4(x3)  # (2, 32, 4, 4)
        x4 = x4+x2  # (2, 32, 4, 4)
        #print(f'x4 shape: {x4.shape}')

        x5 = self.cb5(x4)  # (2, 64, 2, 2)
        #print(f'x5 shape: {x5.shape}')
        x6 = self.cb6(x5)  # (2, 128, 1, 1)
        #print(f'x6 shape: {x6.shape}')

        x7 = self.cb7(x6)  # (2,128,1,1)
        x8 = self.cb8(x7)  # (2,128,1,1)
        #print(f'x8 shape: {x8.shape}')
        x8 = x8+x6

        N, C, H, W = x8.shape
        x8 = x8.reshape((N, C*H*W))  # (2,128)
        x8 = self.linear1(x8)  # (2,128)
        x8 = self.relu(x8)
        x8 = self.linear2(x8)  # (2,10)

        return x8
        # END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        # BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.mode = seq_model
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(
                embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            self.seq_model = nn.LSTM(
                embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size,
                                device=device, dtype=dtype)
        # END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        # BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        
        x = self.embedding(x)  # seq_len, bs, embedding_size
        # x: (seq_len, bs, hidden_size), h: (num_layers, bs, hidden_size)
        x, h = self.seq_model(x, h)
        x = x.reshape((seq_len*bs, self.hidden_size))
        x = self.linear(x)  # (seq_len*bs, output_size)
        return x, h
    

        # END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)
